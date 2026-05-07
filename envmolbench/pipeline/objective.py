"""
Optuna 超参数优化目标函数。

合并自：
    cpu_ml_gnn/objective.py   —— 传统 ML 模型目标函数
    gnn_suite/objective.py    —— GNN 模型目标函数

与原版对齐的核心功能：

1. 特征化分层：
   - 动态特征（morgan/morgan_count）：每次 trial 根据 radius/n_bits 重新生成
   - 静态特征（maccs/mordred）：在 HyperoptObjective.__init__ 中预计算一次，
     所有 trial 共享，避免重复计算

2. 描述符特征筛选（仅 mordred）：
   - 每次 trial 在训练集上 fit DescriptorSelector，
     再用相同列索引处理验证集（防止数据泄漏）

3. 特征缩放：
   - 根据模型类型（SVM/Ridge/Lasso/LogisticRegression）和
     特征类型（mordred/morgan_count）自动决定是否缩放
   - scaler 可以是搜索空间中的超参数（SVM/LogisticRegression）

4. XGBoost/CatBoost 早停：
   - 单次验证（非 CV）时：n_estimators 设为 2000，启用 early_stopping_rounds=50
   - CV 模式：使用搜索空间中给定的迭代次数，不启用早停

5. 双层超时保护（对应原版 config.py 的超时配置）：
   - trial_timeout_seconds（默认 600s / 10分钟）：
     单次 trial 的最大允许时间，通过子进程实现跨平台超时控制。
     超时后该 trial 被标记为极差值，study 继续运行下一个 trial。
   - total_timeout_seconds（默认 7200s / 2小时）：
     整个 study.optimize() 的最大运行时间，直接传给 Optuna。
     用法：study.optimize(objective, timeout=objective.total_timeout_seconds)

6. CUDA OOM 处理（GNN）：
   - 捕获 RuntimeError("out of memory")，返回极差值而不终止 study
"""
import gc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import split_data
from ..models import get_model
from ..common.metrics import calc_metrics
from ..common.utils import apply_feature_scaling, run_with_timeout
from ..common.result_writer import write_hyperopt_iter
from ..config.hyperparams import get_search_space

logger = logging.getLogger(__name__)

# ── 早停配置（对应原版 config.py） ────────────────────────────────────
_DEFAULT_EARLY_STOPPING_ROUNDS = 50
_DEFAULT_N_ESTIMATORS_EARLY_STOP = 2000

# ── 超时默认值（对应原版 config.py） ─────────────────────────────────
# TRIAL_TIMEOUT_SECONDS = 60 * 10  →  单次 trial 超时 10 分钟
# HYPEROPT_TIMEOUT_SECONDS = 360 * 20  →  整体搜索超时 2 小时
_DEFAULT_TRIAL_TIMEOUT = 600     # 10 分钟
_DEFAULT_TOTAL_TIMEOUT = 7200    # 2 小时

# ── 需要强制缩放的模型 ───────────────────────────────────────────────
_MODELS_REQUIRING_SCALING = {"ridge", "lasso", "svr", "svc", "logistic_regression"}
# ── 需要缩放的特征类型 ───────────────────────────────────────────────
_FEATURES_REQUIRING_SCALING = {"mordred", "morgan_count"}

# ── 动态特征化器（每次 trial 可能更换 radius/n_bits） ───────────────
_DYNAMIC_FEATURIZERS = {"morgan", "morgan_count"}
# ── 静态特征化器（预计算一次） ───────────────────────────────────────
_STATIC_FEATURIZERS = {"maccs", "mordred"}


def _needs_scaling(model_name: str, featurizer_name: str,
                   scaler_from_params: Optional[str] = None) -> Optional[str]:
    """
    决定使用哪种 scaler。
    scaler_from_params 是搜索空间中的 'scaler' 参数（SVM/LogisticRegression 时存在）。
    """
    if scaler_from_params:
        return scaler_from_params
    if (model_name.lower() in _MODELS_REQUIRING_SCALING
            or featurizer_name.lower() in _FEATURES_REQUIRING_SCALING):
        return "standard"
    return None


def _is_boosted_tree(model_name: str) -> bool:
    return model_name.lower() in ("xgboost", "xgb", "catboost")


class HyperoptObjective:
    """
    Optuna 超参数优化目标函数封装器。

    支持传统 ML 和 GNN 模型，自动处理：
    - 静态/动态特征化
    - Mordred 描述符特征筛选（per-trial，仅在训练集上 fit）
    - 特征缩放（按模型和特征类型自动决定）
    - XGBoost/CatBoost 早停

    Args:
        model_name:     模型名称（如 'rf', 'xgboost', 'gnn'）。
        smiles:         SMILES 字符串列表。
        labels:         标签列表。
        task_type:      任务类型，'regression' 或 'classification'。
        featurizer:     特征化器名称（'morgan'/'maccs'/'mordred' 等）。
                        传统 ML 必须指定；GNN/Chemprop/ChemBERTa/CNN/UniMol
                        内部自己处理特征化，可传 None。
        split_method:          数据划分方法，默认 'scaffold'。
        n_cv_folds:            交叉验证折数；1 表示单次验证（启用早停）。
        val_size:              验证集比例（n_cv_folds=1 时有效），默认 0.1。
        test_size:             测试集比例，默认 0.1。
        seed:                  随机种子，默认 42。
        metric:                优化目标指标键名；None 时自动选（回归→val_rmse，分类→val_auc_roc）。
        direction:             'minimize' 或 'maximize'。
        extra_config:          额外模型配置（覆盖默认值）。
        trial_timeout_seconds: 单次 trial 最大允许秒数（含全部 CV 折）。
                               默认 600（10 分钟），对应原版 TRIAL_TIMEOUT_SECONDS。
                               超时后该 trial 返回极差值，study 继续。
                               设为 None 则不限制。
        total_timeout_seconds: 整体搜索最大允许秒数。
                               默认 7200（2 小时），对应原版 HYPEROPT_TIMEOUT_SECONDS。
                               通过 study.optimize(timeout=obj.total_timeout_seconds) 使用。
                               设为 None 则不限制。

    用法示例::

        import optuna
        from envmolbench.pipeline.objective import HyperoptObjective

        obj = HyperoptObjective(
            model_name="rf",
            smiles=smiles_list,
            labels=labels_list,
            task_type="regression",
            featurizer="morgan",
            trial_timeout_seconds=600,    # 单次 trial 最长 10 分钟
            total_timeout_seconds=7200,   # 整体最长 2 小时
        )
        study = optuna.create_study(direction=obj.direction)
        # total_timeout_seconds 传给 Optuna 的 timeout 参数
        study.optimize(obj, n_trials=300, timeout=obj.total_timeout_seconds)
        print(study.best_params)
    """

    def __init__(
        self,
        model_name: str,
        smiles: List[str],
        labels: List,
        task_type: str = "regression",
        featurizer: Optional[str] = "morgan",
        split_method: str = "scaffold",
        n_cv_folds: int = 1,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        metric: Optional[str] = None,
        direction: str = "minimize",
        extra_config: Optional[Dict] = None,
        trial_timeout_seconds: Optional[int] = _DEFAULT_TRIAL_TIMEOUT,
        total_timeout_seconds: Optional[int] = _DEFAULT_TOTAL_TIMEOUT,
        trial_log_path: Optional[str] = None,
        data_name: str = "",
    ) -> None:
        self.model_name            = model_name.lower()
        self.smiles                = list(smiles)
        self.labels                = np.asarray(labels)
        self.task_type             = task_type
        self.featurizer            = (featurizer or "").lower()
        self.split_method          = split_method
        self.n_cv_folds            = max(1, n_cv_folds)
        self.val_size              = val_size
        self.test_size             = test_size
        self.seed                  = seed
        self.direction             = direction
        self.extra_config          = extra_config or {}
        self.trial_timeout_seconds = trial_timeout_seconds   # 单次 trial 超时
        self.total_timeout_seconds = total_timeout_seconds   # 整体超时（给 Optuna 用）
        # 超参数迭代日志 CSV（对应原版 hyperopt_iterations_{iter}.csv）
        self.trial_log_path        = trial_log_path
        self.data_name             = data_name or ""

        # 默认优化指标（与原版对齐：验证集前缀使用 valid_ 而非 val_）
        if metric is None:
            if task_type == "regression":
                self.metric    = "valid_rmse"   # 对应原版 v_loss / valid_rmse
                self.direction = "minimize"
            else:
                self.metric    = "valid_auc_roc"
                self.direction = "maximize"
        else:
            self.metric = metric

        # ── 预先划分数据 ──────────────────────────────────────
        # n_cv_folds=1：单次 train/val/test 划分
        # n_cv_folds>1：产生多折，每折含 train_indices, val_indices, test_indices
        self._splits_list: List = self._build_splits()

        # ── 静态特征预计算（maccs/mordred，避免每次 trial 重算） ──
        self._static_X: Optional[np.ndarray] = None     # 全量静态特征矩阵
        self._static_valid_indices: Optional[List[int]] = None  # 有效样本索引
        if self.featurizer in _STATIC_FEATURIZERS:
            self._precompute_static_features()

        logger.info(
            f"[HyperoptObjective] 初始化：model={model_name}, "
            f"task={task_type}, featurizer={featurizer}, "
            f"n_cv_folds={self.n_cv_folds}, metric={self.metric}, "
            f"direction={self.direction}, samples={len(smiles)}, "
            f"trial_timeout={trial_timeout_seconds}s, "
            f"total_timeout={total_timeout_seconds}s"
        )

    # ── 数据划分 ──────────────────────────────────────────────────

    def _build_splits(self) -> List[Dict]:
        """
        构建划分列表。
        每个元素是 {'train_idx', 'val_idx', 'test_idx', 'smiles', 'labels'} 的字典。
        """
        splits = []
        for fold in range(self.n_cv_folds):
            seed_fold = self.seed + fold
            result = split_data(
                self.smiles, self.labels.tolist(),
                method=self.split_method,
                test_size=self.test_size,
                val_size=self.val_size,
                seed=seed_fold,
            )
            splits.append({
                "train_smiles": result.train_smiles,
                "train_labels": np.asarray(result.train_labels),
                "val_smiles":   result.val_smiles,
                "val_labels":   np.asarray(result.val_labels),
                "test_smiles":  result.test_smiles,
                "test_labels":  np.asarray(result.test_labels),
            })
        return splits

    # ── 静态特征预计算 ────────────────────────────────────────────

    def _precompute_static_features(self) -> None:
        """预计算全量静态特征（maccs/mordred），存入 self._static_X。"""
        from ..featurizer import get_featurizer
        logger.info(f"[HyperoptObjective] 预计算静态特征: {self.featurizer}...")
        feat = get_featurizer(self.featurizer)
        X_raw = feat.transform(self.smiles)

        # 记录有效样本（至少有一个非 NaN 的行）
        valid_mask = ~np.isnan(X_raw).all(axis=1)
        self._static_valid_indices = np.where(valid_mask)[0].tolist()
        self._static_X = X_raw[valid_mask].astype(np.float32)

        logger.info(
            f"[HyperoptObjective] 静态特征预计算完成："
            f"有效样本 {len(self._static_valid_indices)}/{len(self.smiles)}，"
            f"原始特征维度 {self._static_X.shape[1]}"
        )

    # ── 单折评估 ──────────────────────────────────────────────────

    def _eval_one_fold(
        self,
        fold: Dict,
        params: Dict,
        use_early_stopping: bool,
    ) -> float:
        """
        在单折数据上完成：特征化 → 特征筛选 → 缩放 → 训练 → 验证集评估。

        Returns:
            验证集目标指标值（float）。
        """
        # ── A. 获取特征矩阵 ────────────────────────────────────
        if self.featurizer in _DYNAMIC_FEATURIZERS:
            # 动态特征：radius/n_bits 是超参数，每次 trial 重算
            from ..featurizer import get_featurizer
            radius = int(params.pop("radius", 2))
            n_bits = int(params.pop("n_bits", 1024))
            feat = get_featurizer(self.featurizer, radius=radius, n_bits=n_bits)
            X_train_raw = feat.transform(fold["train_smiles"]).astype(np.float32)
            X_val_raw   = feat.transform(fold["val_smiles"]).astype(np.float32)

            # 移除全零列（对应原版 remove_all_zero_columns）
            nonzero_mask = (X_train_raw != 0).any(axis=0)
            X_train_raw = X_train_raw[:, nonzero_mask]
            X_val_raw   = X_val_raw[:, nonzero_mask]

        else:
            # 静态特征：从预计算矩阵中按索引提取
            # 需要把 fold 中的 smiles 索引映射回全量索引
            valid_set = set(self._static_valid_indices)
            # 全量 smiles → index 映射
            smiles_to_idx = {s: i for i, s in enumerate(self.smiles)}

            # 构建全量有效索引的逆查找表（orig_idx → 矩阵行号），避免 O(n) 的 list.index()
            valid_idx_to_row = {orig_idx: row_i
                                for row_i, orig_idx in enumerate(self._static_valid_indices)}

            def _get_rows(smiles_list, labels_arr):
                """从静态特征矩阵中提取对应行，同步过滤标签。"""
                rows, labels_keep = [], []
                for smi, lbl in zip(smiles_list, labels_arr):
                    orig_idx = smiles_to_idx.get(smi)
                    if orig_idx is not None and orig_idx in valid_set:
                        row_i = valid_idx_to_row[orig_idx]
                        rows.append(self._static_X[row_i])
                        labels_keep.append(lbl)
                return (np.vstack(rows).astype(np.float32) if rows else
                        np.empty((0, self._static_X.shape[1]), dtype=np.float32),
                        np.asarray(labels_keep))

            X_train_raw, y_train = _get_rows(fold["train_smiles"], fold["train_labels"])
            X_val_raw,   y_val   = _get_rows(fold["val_smiles"],   fold["val_labels"])
            fold = {**fold, "train_labels": y_train, "val_labels": y_val}

        y_train = fold["train_labels"]
        y_val   = fold["val_labels"]

        if X_train_raw.shape[0] == 0:
            raise ValueError("训练集特征化后为空，跳过此折。")

        # ── B. Mordred 描述符特征筛选 ─────────────────────────
        if self.featurizer == "mordred":
            from ..featurizer.descriptor import DescriptorSelector
            selector = DescriptorSelector(
                correlation_threshold=0.8,
                variance_threshold=0.0,
            )
            X_train_proc = selector.fit_transform(X_train_raw)
            X_val_proc   = selector.transform(X_val_raw)
            logger.debug(
                f"  [Mordred筛选] {X_train_raw.shape[1]} → {X_train_proc.shape[1]} 维"
            )
        else:
            X_train_proc = X_train_raw
            X_val_proc   = X_val_raw

        # ── C. 特征缩放 ────────────────────────────────────────
        # 搜索空间中的 scaler 参数（SVM/LogisticRegression 专用）
        scaler_from_params = params.pop("scaler", None)
        scaler_type = _needs_scaling(self.model_name, self.featurizer, scaler_from_params)
        X_train, X_val, _, _ = apply_feature_scaling(
            X_train_proc, X_val_proc, None, scaler_type
        )

        # ── D. 构建 sklearn 模型并训练 ──────────────────────────
        # 使用 ClassicalModel 的 _build_model 逻辑，但直接操作特征矩阵
        from ..models.classical import ClassicalModel
        clf = ClassicalModel(
            model_name=self.model_name,
            task_type=self.task_type,
            model_params=params,
        )
        sklearn_model = clf._build_model()

        fit_params: Dict = {}
        if use_early_stopping and _is_boosted_tree(self.model_name):
            if self.model_name in ("xgboost", "xgb"):
                sklearn_model.set_params(
                    n_estimators=_DEFAULT_N_ESTIMATORS_EARLY_STOP,
                    early_stopping_rounds=_DEFAULT_EARLY_STOPPING_ROUNDS,
                )
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"]  = False
            elif self.model_name == "catboost":
                sklearn_model.set_params(
                    iterations=_DEFAULT_N_ESTIMATORS_EARLY_STOP,
                )
                fit_params["eval_set"]              = (X_val, y_val)
                fit_params["early_stopping_rounds"] = _DEFAULT_EARLY_STOPPING_ROUNDS
                fit_params["verbose"]               = False

        sklearn_model.fit(X_train, y_train, **fit_params)

        # ── E. 验证集评估 ──────────────────────────────────────
        if self.task_type == "classification":
            if hasattr(sklearn_model, "predict_proba"):
                val_preds = sklearn_model.predict_proba(X_val)[:, 1]
            else:
                val_preds = sklearn_model.decision_function(X_val)
        else:
            val_preds = sklearn_model.predict(X_val)

        # 使用 valid 前缀（与原版 evaluator.py 一致：valid_rmse / valid_auc_roc）
        metrics = calc_metrics(
            y_val.tolist(), val_preds.tolist(),
            task=self.task_type,
            prefix="valid",
        )
        score = metrics.get(self.metric)
        if score is None:
            raise ValueError(
                f"指标 '{self.metric}' 不在评估结果中：{list(metrics.keys())}"
            )
        return float(score)

    # ── Optuna __call__ ──────────────────────────────────────────

    def __call__(self, trial) -> float:
        """
        Optuna 调用接口：采样超参数 → 单折/多折评估 → 返回验证集均值。

        n_cv_folds=1：单次验证，XGB/CatBoost 启用早停。
        n_cv_folds>1：交叉验证，XGB/CatBoost 不启用早停（使用固定迭代次数）。

        单次 trial 超时由 trial_timeout_seconds 控制（默认 10 分钟）；
        整体超时由 study.optimize(timeout=obj.total_timeout_seconds) 控制。
        """
        params = get_search_space(
            self.model_name, trial,
            task_type=self.task_type,
            featurizer=self.featurizer,
        )
        # 合并额外固定配置（用户显式指定的参数优先级高于搜索空间）
        params = {**params, **self.extra_config}

        # 单次验证启用早停，CV 不启用
        use_early_stopping = (self.n_cv_folds == 1)

        # 仅提取 trial.number（int），避免闭包捕获不可 pickle 的 Optuna Trial 对象
        # （Windows 使用 spawn 启动子进程时，所有捕获对象都需要可被 pickle 序列化）
        trial_number = trial.number
        logger.debug(f"[Trial {trial_number}] 超参数: {params}")

        # ── 单次 trial 超时包装 ────────────────────────────────
        # 所有折的评估打包为一个函数，整体受 trial_timeout_seconds 限制。
        # 使用子进程实现跨平台超时（Windows/Linux 均有效）。
        # 注意：闭包只捕获可 pickle 的对象（self、params、use_early_stopping、trial_number），
        #        不捕获 Optuna trial 对象本身，以保证 Windows 兼容性。
        def _run_all_folds() -> float:
            """在（可能的）子进程中执行全部折评估，返回均值。"""
            fold_scores = []
            for fold_idx, fold in enumerate(self._splits_list):
                score = self._eval_one_fold(
                    fold, params.copy(), use_early_stopping
                )
                fold_scores.append(score)
                logger.debug(
                    f"[Trial {trial_number}] Fold {fold_idx+1}/{self.n_cv_folds}: "
                    f"{self.metric}={score:.6f}"
                )
            return float(np.mean(fold_scores))

        bad_score = float("inf") if self.direction == "minimize" else float("-inf")

        try:
            if self.trial_timeout_seconds is not None:
                # 通过子进程实现超时（对应原版 run_with_timeout）
                final_score = run_with_timeout(
                    _run_all_folds,
                    args=(),
                    kwargs={},
                    timeout_seconds=self.trial_timeout_seconds,
                )
            else:
                final_score = _run_all_folds()

        except TimeoutError:
            logger.warning(
                f"[Trial {trial_number}] 超时 (>{self.trial_timeout_seconds}s)，"
                f"返回极差值。"
            )
            if self.trial_log_path:
                write_hyperopt_iter(self.trial_log_path, self.data_name,
                                    self.model_name, self.featurizer,
                                    bad_score, params, trial_number, "timeout")
            return bad_score
        except Exception as exc:
            err_str = str(exc).lower()
            if "out of memory" in err_str:
                logger.warning(f"[Trial {trial_number}] CUDA OOM，返回极差值。")
                status = "oom"
            else:
                logger.warning(f"[Trial {trial_number}] 失败: {exc}")
                status = f"error: {type(exc).__name__}"
            if self.trial_log_path:
                write_hyperopt_iter(self.trial_log_path, self.data_name,
                                    self.model_name, self.featurizer,
                                    bad_score, params, trial_number, status)
            return bad_score
        finally:
            gc.collect()

        logger.info(
            f"[Trial {trial_number}] 平均 {self.metric}={final_score:.6f}  "
            f"params={params}"
        )

        # ── 写入单次 trial 迭代日志（对应原版 hyperopt_iterations_csv） ──
        if self.trial_log_path:
            try:
                write_hyperopt_iter(
                    log_csv=self.trial_log_path,
                    data_name=self.data_name,
                    model_name=self.model_name,
                    featuring=self.featurizer,
                    v_loss=final_score,
                    params=params,
                    iteration=trial_number,
                    status="ok",
                )
            except Exception as log_exc:
                logger.warning(f"写入 trial 日志失败: {log_exc}")

        return final_score

    # ── 最优参数重训 ──────────────────────────────────────────────

    def best_params_retrain(self, best_params: Dict) -> Dict:
        """
        用最优超参数在 train+val 上重训，对 train/valid/test 三个子集评估并返回指标。

        对应原版 WorkflowRunner._final_evaluate_and_save 的逻辑。
        使用第一折数据：train+val 合并训练，test 集最终评估。

        Args:
            best_params: optuna study.best_params 字典。

        Returns:
            包含 train/valid/test 全量指标的结果字典，
            以及 'model'、'featurizer'、'best_params'、'_predictions' 键。
            '_predictions' 包含逐样本预测数据，供调用方保存预测文件。
        """
        logger.info(f"[HyperoptObjective] 最优参数重训: {best_params}")
        fold = self._splits_list[0]
        combined_smiles = fold["train_smiles"] + fold["val_smiles"]
        combined_labels = np.concatenate(
            [fold["train_labels"], fold["val_labels"]]
        ).tolist()

        params = {**best_params, **self.extra_config}

        _is_dl_model = self.model_name in (
            "gnn", "gcn", "chemprop", "chemberta", "cnn", "unimol"
        )

        if _is_dl_model or self.featurizer not in (
            _DYNAMIC_FEATURIZERS | _STATIC_FEATURIZERS | {""}
        ):
            # 深度学习模型：通过 get_model 接口训练，内部自行特征化
            model = get_model(self.model_name, task_type=self.task_type)
            model.fit(combined_smiles, combined_labels,
                      fold["val_smiles"], fold["val_labels"].tolist())
        else:
            # 传统 ML：ClassicalModel 完整流程（含特征筛选和缩放）
            scaler_type = params.pop("scaler", None)
            radius = int(params.pop("radius", 2))
            n_bits = int(params.pop("n_bits", 1024))
            from ..featurizer import get_featurizer
            if self.featurizer in _DYNAMIC_FEATURIZERS:
                feat = get_featurizer(self.featurizer, radius=radius, n_bits=n_bits)
            else:
                feat = get_featurizer(self.featurizer)
            model = get_model(
                self.model_name,
                task_type=self.task_type,
                featurizer=feat,
                scaler_type=scaler_type,
                model_params=params,
            )
            model.fit(combined_smiles, combined_labels,
                      fold["val_smiles"], fold["val_labels"].tolist())

        # 在三个子集上评估（使用 valid 前缀与原版对齐）
        all_metrics = {}
        predictions = {}
        for subset_name, smiles_list, true_labels in [
            ("train", fold["train_smiles"],  fold["train_labels"]),
            ("valid", fold["val_smiles"],     fold["val_labels"]),
            ("test",  fold["test_smiles"],    fold["test_labels"]),
        ]:
            if not smiles_list:
                continue
            preds = model.predict(smiles_list)
            m = calc_metrics(
                np.asarray(true_labels).tolist(), preds.tolist(),
                task=self.task_type, prefix=subset_name,
            )
            all_metrics.update(m)
            predictions[subset_name] = {
                "smiles":     smiles_list,
                "true_label": true_labels,
                "predicted":  preds,
            }

        all_metrics["model"]        = self.model_name
        all_metrics["featurizer"]   = self.featurizer
        all_metrics["best_params"]  = best_params
        all_metrics["_predictions"] = predictions   # 供 runner 保存预测文件
        return all_metrics
