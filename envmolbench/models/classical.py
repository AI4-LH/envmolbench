"""
传统机器学习模型封装。

迁移自 cpu_ml_gnn/model_wrapper.py 的 ModelWrapper 类，
重构为继承 BaseModel 的 ClassicalModel，
统一接口：fit(smiles, labels) → 内部自动特征化 → 特征筛选（Mordred专用）→ 特征缩放 → 训练 sklearn 模型。

特征筛选规则（仅 MordredFeaturizer）：
    1. 移除含 NaN/Inf 的列
    2. 低方差过滤（零方差列）
    3. 高相关性过滤（相关系数 > 0.8，保留方差大的）

需要自动缩放的模型（对应原版 MODELS_REQUIRING_SCALING）：
    Ridge, Lasso, SVR, SVC, LogisticRegression

需要缩放的特征类型（对应原版 FEATURES_REQUIRING_SCALING）：
    mordred（Molecular_Descriptors），morgan_count（Morgan_count）
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseModel
from ..featurizer import get_featurizer, BaseFeaturizer
from ..common.utils import apply_feature_scaling

logger = logging.getLogger(__name__)

# ── 需要强制缩放的模型（无论特征类型） ────────────────────────────────
_MODELS_REQUIRING_SCALING = {"ridge", "lasso", "svr", "svc", "logistic_regression"}

# ── 需要缩放的特征类型（无论模型） ────────────────────────────────────
_FEATURES_REQUIRING_SCALING = {"mordred", "morgan_count"}

# ── 模型名称 → sklearn/xgboost/catboost 类的映射（延迟导入） ───────────
_REGRESSION_MODELS = {
    "ridge":          ("sklearn.linear_model", "Ridge"),
    "lasso":          ("sklearn.linear_model", "Lasso"),
    "svr":            ("sklearn.svm", "SVR"),
    "rf":             ("sklearn.ensemble", "RandomForestRegressor"),
    "random_forest":  ("sklearn.ensemble", "RandomForestRegressor"),
    "xgboost":        ("xgboost", "XGBRegressor"),
    "xgb":            ("xgboost", "XGBRegressor"),
    "catboost":       ("catboost", "CatBoostRegressor"),
}

_CLASSIFICATION_MODELS = {
    "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
    "svc":                 ("sklearn.svm", "SVC"),
    "rf":                  ("sklearn.ensemble", "RandomForestClassifier"),
    "random_forest":       ("sklearn.ensemble", "RandomForestClassifier"),
    "xgboost":             ("xgboost", "XGBClassifier"),
    "xgb":                 ("xgboost", "XGBClassifier"),
    "catboost":            ("catboost", "CatBoostClassifier"),
}

# ── 早停默认配置（对应原版 config.py） ────────────────────────────────
_DEFAULT_EARLY_STOPPING_ROUNDS = 50
_DEFAULT_N_ESTIMATORS_EARLY_STOP = 2000


def _import_model_class(module_name: str, class_name: str):
    """动态导入模型类，返回类对象。"""
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def _needs_scaling(model_name: str, featurizer_name: str) -> bool:
    """判断该模型-特征组合是否需要特征缩放。"""
    return (
        model_name.lower() in _MODELS_REQUIRING_SCALING
        or featurizer_name.lower() in _FEATURES_REQUIRING_SCALING
    )


class ClassicalModel(BaseModel):
    """
    传统机器学习模型封装类。

    内部流程：
        SMILES → 特征化
               → （仅 Mordred）DescriptorSelector 特征筛选
               → （按需）特征缩放
               → sklearn/XGB/CatBoost 模型训练/推理

    Args:
        model_name:   模型名称（如 'rf'、'xgboost'、'ridge'、'svc'）。
        task_type:    'regression' 或 'classification'。
        featurizer:   特征化方法名称或 BaseFeaturizer 实例，默认 'morgan'。
        scaler_type:  特征缩放类型（'standard'/'minmax'/'maxabs'/'robust'/None）。
                      None 时根据模型和特征类型自动决定：
                      Ridge/Lasso/SVR/SVC/LogisticRegression → 'standard'；
                      Mordred/MorganCount 特征 → 'standard'。
        model_params: 直接传给 sklearn 模型的参数字典。
        config:       从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        model_name: str = "rf",
        task_type: str = "regression",
        featurizer: Union[str, BaseFeaturizer] = "morgan",
        scaler_type: Optional[str] = None,   # None = 自动判断
        model_params: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.model_name = model_name.lower()
        self._scaler_type_override = scaler_type   # 保存用户显式设置值
        self.model_params = model_params or {}

        # 初始化特征化器
        self._featurizer: BaseFeaturizer = (
            featurizer if isinstance(featurizer, BaseFeaturizer)
            else get_featurizer(featurizer)
        )
        self._model = None
        self._fitted_scaler = None
        self._descriptor_selector = None   # 仅 MordredFeaturizer 使用

    # ── 内部工具 ──────────────────────────────────────────────────

    def _get_effective_scaler_type(self) -> Optional[str]:
        """自动决定缩放类型：用户覆盖 > 自动规则 > None。"""
        if self._scaler_type_override is not None:
            return self._scaler_type_override
        feat_name = getattr(self._featurizer, "name", "")
        if _needs_scaling(self.model_name, feat_name):
            return "standard"
        return None

    def _is_mordred_featurizer(self) -> bool:
        """判断当前特征化器是否是 MordredFeaturizer（需要特征筛选）。"""
        from ..featurizer.descriptor import MordredFeaturizer
        return isinstance(self._featurizer, MordredFeaturizer)

    def _build_model(self) -> Any:
        """根据 model_name 和 task_type 实例化 sklearn 模型。"""
        model_map = (
            _REGRESSION_MODELS if self.task_type == "regression" else _CLASSIFICATION_MODELS
        )
        if self.model_name not in model_map:
            available = sorted(model_map.keys())
            raise ValueError(f"未知模型 '{self.model_name}'，可用: {available}")

        module_name, class_name = model_map[self.model_name]
        model_cls = _import_model_class(module_name, class_name)

        # 通用默认参数（对应原版 ModelWrapper._create_new_instance）
        default_params: Dict[str, Any] = {}
        # SVR/SVC 没有 random_state 参数
        if self.model_name not in ("svr", "svc"):
            default_params["random_state"] = self.config.get("seed", 42)
        if self.model_name == "svc":
            default_params["probability"] = True   # 支持 predict_proba
        if self.model_name in ("rf", "random_forest"):
            default_params["n_jobs"] = -1
        if self.model_name in ("svr", "svc", "logistic_regression", "lasso"):
            default_params["max_iter"] = 10000     # 保证收敛

        # XGBoost 默认配置
        if self.model_name in ("xgboost", "xgb"):
            default_params.update({
                "tree_method": "hist",
                "verbosity": 0,
                "n_estimators": _DEFAULT_N_ESTIMATORS_EARLY_STOP,
            })
            if self.task_type == "classification":
                default_params.update({
                    "use_label_encoder": False,
                    "n_jobs": -1,
                    "eval_metric": "logloss",
                })
            else:
                default_params["eval_metric"] = "rmse"

        # CatBoost 默认配置
        if self.model_name == "catboost":
            default_params.update({
                "task_type": "CPU",
                "iterations": _DEFAULT_N_ESTIMATORS_EARLY_STOP,
                "verbose": False,
            })
            if self.task_type == "regression":
                default_params.update({"loss_function": "RMSE", "eval_metric": "RMSE"})
            else:
                default_params.update({"loss_function": "Logloss", "eval_metric": "AUC"})

        # 过滤掉模型不支持的参数，再合并用户参数
        try:
            dummy = model_cls(**default_params)
            valid_keys = set(dummy.get_params().keys())
            final_params = {
                k: v for k, v in {**default_params, **self.model_params}.items()
                if k in valid_keys
            }
        except Exception:
            # 如果 dummy 初始化失败，直接合并（模型自己报错）
            final_params = {**default_params, **self.model_params}

        logger.debug(f"创建 {class_name}，参数: {final_params}")
        return model_cls(**final_params)

    # ── fit ──────────────────────────────────────────────────────

    def fit(
        self,
        train_smiles: List[str],
        train_labels,
        val_smiles: Optional[List[str]] = None,
        val_labels=None,
    ) -> "ClassicalModel":
        """
        完整训练流程：特征化 → 特征筛选（Mordred）→ 特征缩放 → 模型训练。

        XGBoost/CatBoost 当提供验证集时自动启用早停。
        """
        logger.info(
            f"[ClassicalModel] 训练 {self.model_name}（{self.task_type}），"
            f"特征化: {getattr(self._featurizer, 'name', '?')}，"
            f"样本数: {len(train_smiles)}"
        )
        train_labels = np.asarray(train_labels)

        # ── 1. 特征化 ──────────────────────────────────────────
        X_train_raw = self._featurizer.transform(train_smiles)
        X_val_raw = self._featurizer.transform(val_smiles) if val_smiles else None

        # 清理无效 SMILES（整行全 NaN）
        valid_mask = ~np.isnan(X_train_raw).all(axis=1)
        if valid_mask.sum() < len(train_smiles):
            logger.warning(
                f"  训练集 {len(train_smiles) - valid_mask.sum()} 个 SMILES 特征化失败，已剔除。"
            )
        X_train_raw = X_train_raw[valid_mask]
        y_train = train_labels[valid_mask]

        # ── 2. 描述符特征筛选（仅 Mordred） ───────────────────
        if self._is_mordred_featurizer():
            from ..featurizer.descriptor import DescriptorSelector
            logger.info("  [Mordred] 执行特征筛选（NaN过滤 + 方差过滤 + 相关性过滤）...")
            self._descriptor_selector = DescriptorSelector(
                correlation_threshold=0.8,
                variance_threshold=0.0,
            )
            X_train_proc = self._descriptor_selector.fit_transform(X_train_raw)
            # 验证筛选效果
            self._descriptor_selector.verify_correlation(X_train_proc)
            # 同样的列索引应用到验证集
            X_val_proc = (
                self._descriptor_selector.transform(X_val_raw)
                if X_val_raw is not None else None
            )
        else:
            X_train_proc = X_train_raw
            X_val_proc = X_val_raw

        # ── 3. 特征缩放 ────────────────────────────────────────
        scaler_type = self._get_effective_scaler_type()
        X_train, X_val_scaled, _, self._fitted_scaler = apply_feature_scaling(
            X_train_proc, X_val_proc, None, scaler_type
        )
        if scaler_type:
            logger.info(f"  特征缩放: {scaler_type}")

        # ── 4. 构建并训练模型 ──────────────────────────────────
        self._model = self._build_model()
        fit_params: Dict[str, Any] = {}

        # XGBoost/CatBoost 早停（仅单次训练，有验证集时）
        val_labels_arr = np.asarray(val_labels) if val_labels is not None else None
        if X_val_scaled is not None and val_labels_arr is not None:
            if self.model_name in ("xgboost", "xgb"):
                fit_params["eval_set"] = [(X_val_scaled, val_labels_arr)]
                fit_params["verbose"] = False
                # early_stopping_rounds 通过 model_params 或默认值设置
                if "early_stopping_rounds" not in self.model_params:
                    self._model.set_params(
                        early_stopping_rounds=_DEFAULT_EARLY_STOPPING_ROUNDS
                    )
            elif self.model_name == "catboost":
                fit_params["eval_set"] = (X_val_scaled, val_labels_arr)
                fit_params["early_stopping_rounds"] = self.model_params.get(
                    "early_stopping_rounds", _DEFAULT_EARLY_STOPPING_ROUNDS
                )
                fit_params["verbose"] = False

        self._model.fit(X_train, y_train, **fit_params)
        self._is_fitted = True
        logger.info(
            f"[ClassicalModel] {self.model_name} 训练完成，"
            f"特征维度: {X_train.shape[1]}"
        )
        return self

    # ── predict ──────────────────────────────────────────────────

    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        推理流程：特征化 → 特征筛选（Mordred）→ 特征缩放 → 预测。
        无法解析的 SMILES 对应位置填 NaN。
        """
        self._check_fitted()
        X_raw = self._featurizer.transform(smiles)

        # 无效 SMILES 位置记录（整行全 NaN）
        valid_mask = ~np.isnan(X_raw).all(axis=1)
        result = np.full(len(smiles), np.nan, dtype=np.float32)

        X_valid = X_raw[valid_mask]
        if X_valid.shape[0] == 0:
            return result

        # 描述符特征筛选（用 fit 时保存的列索引）
        if self._descriptor_selector is not None:
            X_valid = self._descriptor_selector.transform(X_valid)

        # 特征缩放
        if self._fitted_scaler is not None:
            X_valid = self._fitted_scaler.transform(X_valid)

        # 预测
        if self.task_type == "classification":
            preds = self._model.predict_proba(X_valid)
            if preds.ndim == 2:
                preds = preds[:, 1]
        else:
            preds = self._model.predict(X_valid)

        result[valid_mask] = preds.astype(np.float32)
        return result

    # ── save / load ───────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """保存模型、特征化器、缩放器和描述符选择器到 pkl 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":               self._model,
            "featurizer":          self._featurizer,
            "scaler":              self._fitted_scaler,
            "descriptor_selector": self._descriptor_selector,
            "task_type":           self.task_type,
            "model_name":          self.model_name,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"模型已保存到 {path}")

    def load(self, path: Union[str, Path]) -> "ClassicalModel":
        """从 pkl 文件加载模型（包含特征化器、缩放器、描述符选择器）。"""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model               = payload["model"]
        self._featurizer          = payload["featurizer"]
        self._fitted_scaler       = payload["scaler"]
        self._descriptor_selector = payload.get("descriptor_selector")
        self.task_type            = payload["task_type"]
        self.model_name           = payload["model_name"]
        self._is_fitted = True
        logger.info(f"模型已从 {path} 加载。")
        return self

    @property
    def name(self) -> str:
        return f"classical_{self.model_name}"
