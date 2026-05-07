"""
流程编排器（原 WorkflowRunner 重命名为 PipelineRunner）。

提供两种训练-评估流程：
1. run()          —— 使用固定超参数直接训练（快速评估/基线）
2. run_hyperopt() —— Optuna 超参数搜索 + 最优参数重训（工业级调参）

输出文件与原版 cpu_ml_gnn/main_runner.py 对齐：
    {results_dir}/{dataset}/
        baseline_metrics_{split}_{cv_mode}.csv         —— 固定参数基线评估结果
        best_hyperparameters_{split}_{cv_mode}.csv      —— 超参数搜索最优结果
        hyperopt_iterations.csv                         —— 每次 trial 的搜索记录
        predictions/
            {model}_{feat}_{split}_predictions_{data}.csv —— 逐分子预测结果

CSV 列名规范（与原版对齐）：
    - 验证集指标前缀：valid_*（原版使用 valid_，非 val_）
    - 分类 F1 键名：f1_score（原版使用 f1_score，非 f1）
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..data import load_dataset, split_data
from ..models import get_model, BaseModel
from ..common.metrics import calc_metrics
from ..common.result_writer import (
    write_result,
    write_predictions,
    get_result_header,
    is_experiment_done,
)
from ..common.logger import get_logger

logger = logging.getLogger(__name__)


def _make_early_stop_callback(patience: int, direction: str):
    """
    创建 Optuna 早停回调函数。

    连续 `patience` 个 trial 没有改进时停止搜索，
    对应原版 DEFAULT_HYPEROPT_EARLY_STOP=30。

    Args:
        patience:  最大无改进 trial 数。
        direction: 'minimize' 或 'maximize'。

    Returns:
        可传入 study.optimize(callbacks=[...]) 的回调函数。
    """
    no_improve_count = [0]
    best_value = [float("inf") if direction == "minimize" else float("-inf")]

    def callback(study, trial) -> None:
        current = trial.value
        if current is None:
            return
        improved = (
            current < best_value[0] if direction == "minimize"
            else current > best_value[0]
        )
        if improved:
            best_value[0] = current
            no_improve_count[0] = 0
        else:
            no_improve_count[0] += 1
            if no_improve_count[0] >= patience:
                logger.info(
                    f"[EarlyStop] 连续 {patience} 个 trial 无改进，提前停止搜索。"
                    f"当前最优值：{study.best_value:.6f}"
                )
                study.stop()

    return callback


class PipelineRunner:
    """
    标准训练-评估流程编排器。

    负责将数据加载、分割、训练、评估、结果保存串联为完整流程，
    供 CLI 和批量运行使用。

    Args:
        config:      全局配置字典（由 config_loader.load_config 合并而来）。
        result_csv:  基线/最优参数汇总 CSV 路径；None 时不写入文件。
        results_dir: 输出根目录，用于组织预测文件和日志；None 时不保存预测。
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        result_csv: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config or {}
        self.result_csv  = Path(result_csv)  if result_csv  else None
        self.results_dir = Path(results_dir) if results_dir else None

    # ── 标准固定参数训练流程 ─────────────────────────────────────────

    def run(
        self,
        model_name: str,
        dataset_name: str,
        featurizer: Optional[str] = None,
        split_method: Optional[str] = None,
        task_type: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        save_predictions: bool = True,
    ) -> Dict:
        """
        执行单次固定参数训练-评估流程（对应原版 run_baseline_evaluation 的单模型版本）。

        Args:
            model_name:        模型名称（如 'chemberta'、'rf'）。
            dataset_name:      数据集名称或 CSV 路径。
            featurizer:        特征化器名称（传统 ML 时写入结果文件的 featuring 列）。
            split_method:      数据划分方法；None 时读取 config。
            task_type:         任务类型；None 时自动检测。
            datasets_dir:      数据集目录；None 时使用默认路径。
            save_predictions:  是否保存逐样本预测 CSV，默认 True。

        Returns:
            包含 train/valid/test 指标的结果字典。
        """
        split = split_method or self.config.get("training", {}).get("split_method", "scaffold")
        seed  = self.config.get("training", {}).get("seed", 42)
        test_size = self.config.get("training", {}).get("test_size", 0.1)
        val_size  = self.config.get("training", {}).get("val_size",  0.1)

        logger.info(
            f"[PipelineRunner.run] 开始：model={model_name}，"
            f"dataset={dataset_name}，split={split}"
        )

        # 1. 加载数据集
        smiles, labels, detected_task = load_dataset(
            dataset_name, task_type=task_type, datasets_dir=datasets_dir,
        )
        task = detected_task

        # 2. 数据划分（支持 pkl 缓存）
        cache_path = self._split_cache_path(dataset_name, split, cv_mode="single")
        splits = split_data(
            smiles, labels,
            method=split, test_size=test_size, val_size=val_size,
            seed=seed, cache_path=cache_path,
        )

        # 3. 初始化并训练模型
        model = get_model(model_name, task_type=task)
        model.fit(
            splits.train_smiles, splits.train_labels,
            splits.val_smiles,   splits.val_labels,
        )

        # 4. 评估（使用 valid 前缀，与原版对齐）
        results = self._evaluate(model, splits, task)
        results.update({
            "data":       dataset_name,
            "model":      model_name,
            "featuring":  featurizer or "",
            "split_type": split,
            "cv_mode":    "single",
            "task":       task,
        })

        # 5. 保存逐样本预测 CSV
        if save_predictions and self.results_dir:
            self._save_predictions(
                model, splits, task, model_name, featurizer or "", split, dataset_name
            )

        # 6. 写入汇总结果 CSV（_predictions 不写入，仅用于内部传递）
        results.pop("_predictions", None)
        if self.result_csv:
            header = get_result_header(task, hyperopt=False)
            write_result(self.result_csv, results, header)

        logger.info(
            f"[PipelineRunner.run] 完成："
            f"test={results.get('test_rmse', results.get('test_auc_roc', 'N/A'))}"
        )
        return results

    # ── Optuna 超参数搜索流程 ─────────────────────────────────────────

    def run_hyperopt(
        self,
        model_name: str,
        dataset_name: str,
        featurizer: Optional[str] = None,
        n_cv_folds: int = 1,
        split_method: Optional[str] = None,
        task_type: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        n_trials: Optional[int] = None,
        trial_timeout_seconds: Optional[int] = None,
        total_timeout_seconds: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict:
        """
        Optuna 超参数搜索 + 最优参数重训流程。

        流程：
            1. 加载数据集并自动识别任务类型
            2. 断点续跑检查（已完成则直接返回 None）
            3. 构建 HyperoptObjective（含静态特征预计算、折划分）
            4. 创建 Optuna study 并调用 study.optimize()
            5. 用 study.best_params 重训（train+val），在测试集上评估
            6. 保存预测文件和汇总结果

        Args:
            model_name:            模型名称（如 'rf'、'xgboost'、'gnn'）。
            dataset_name:          数据集名称或 CSV 路径。
            featurizer:            特征化器名称（传统 ML 必须指定，如 'morgan'/'mordred'）。
            n_cv_folds:            交叉验证折数；1 = 单次验证（启用 XGB/CatBoost 早停）。
            split_method:          数据划分方法；None 时读取 config。
            task_type:             任务类型；None 时自动检测。
            datasets_dir:          数据集目录；None 时使用默认路径。
            n_trials:              最大 trial 次数；None 时读取 config["hyperopt"]["max_trials"]。
            trial_timeout_seconds: 单次 trial 最大秒数；None 时读取 config["hyperopt"]["trial_timeout_seconds"]。
            total_timeout_seconds: 整体搜索最大秒数；None 时读取 config["hyperopt"]["total_timeout_seconds"]。
            save_predictions:      是否保存逐样本预测 CSV，默认 True。

        Returns:
            包含测试集指标、最优超参数、搜索元信息的结果字典。
            若已完成（断点续跑跳过）则返回 {'skipped': True}。
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("超参数搜索需要 optuna，请运行：pip install optuna")

        from .objective import HyperoptObjective

        # ── 读取超参数搜索配置 ────────────────────────────────────
        hp_cfg = self.config.get("hyperopt", {})
        _n_trials       = n_trials or hp_cfg.get("max_trials", 300)
        _trial_timeout  = trial_timeout_seconds
        if _trial_timeout is None:
            _trial_timeout = hp_cfg.get("trial_timeout_seconds", 600)
        _total_timeout  = total_timeout_seconds
        if _total_timeout is None:
            _total_timeout = hp_cfg.get("total_timeout_seconds", 7200)

        # ── 读取训练配置 ──────────────────────────────────────────
        split     = split_method or self.config.get("training", {}).get("split_method", "scaffold")
        seed      = self.config.get("training", {}).get("seed", 42)
        test_size = self.config.get("training", {}).get("test_size", 0.1)
        val_size  = self.config.get("training", {}).get("val_size",  0.1)
        cv_mode   = f"cv{n_cv_folds}" if n_cv_folds > 1 else "single"

        logger.info(
            f"[PipelineRunner.run_hyperopt] 开始：model={model_name}, "
            f"dataset={dataset_name}, featurizer={featurizer}, "
            f"n_cv_folds={n_cv_folds}, n_trials={_n_trials}, "
            f"trial_timeout={_trial_timeout}s, total_timeout={_total_timeout}s"
        )

        # ── 断点续跑：已有结果则跳过 ─────────────────────────────
        if self.result_csv and self.result_csv.exists():
            if is_experiment_done(
                self.result_csv, model_name, dataset_name, split,
                featuring=featurizer or "", cv_mode=cv_mode
            ):
                logger.info(
                    f"[PipelineRunner.run_hyperopt] 已跳过（结果已存在）："
                    f"{model_name}/{featurizer}/{dataset_name}/{split}/{cv_mode}"
                )
                return {"skipped": True}

        # ── 1. 加载数据集 ──────────────────────────────────────────
        smiles, labels, detected_task = load_dataset(
            dataset_name, task_type=task_type, datasets_dir=datasets_dir,
        )
        task = detected_task

        # ── 超参数迭代日志路径（对应原版 hyperopt_iterations_csv） ──
        trial_log_path = None
        if self.results_dir:
            data_dir = self.results_dir / dataset_name
            data_dir.mkdir(parents=True, exist_ok=True)
            trial_log_path = str(data_dir / "hyperopt_iterations.csv")

        # ── 2. 构建目标函数（含静态特征预计算、折划分） ──────────────
        objective = HyperoptObjective(
            model_name=model_name,
            smiles=smiles,
            labels=labels,
            task_type=task,
            featurizer=featurizer or "",
            split_method=split,
            n_cv_folds=n_cv_folds,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            trial_timeout_seconds=_trial_timeout,
            total_timeout_seconds=_total_timeout,
            trial_log_path=trial_log_path,
            data_name=dataset_name,
        )

        # ── 3. 创建 Optuna study 并搜索 ───────────────────────────
        study = optuna.create_study(
            direction=objective.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        early_stop_trials = hp_cfg.get("early_stop_trials", 30)
        callbacks = []
        if early_stop_trials and early_stop_trials > 0:
            callbacks.append(_make_early_stop_callback(early_stop_trials, objective.direction))

        logger.info(
            f"[PipelineRunner.run_hyperopt] 启动 Optuna study "
            f"（direction={objective.direction}, n_trials={_n_trials}, "
            f"timeout={_total_timeout}s, early_stop_trials={early_stop_trials}）"
        )
        study.optimize(
            objective,
            n_trials=_n_trials,
            timeout=_total_timeout if _total_timeout else None,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        # ── 4. 最优参数重训并评估测试集 ────────────────────────────
        best_params = study.best_params
        logger.info(
            f"[PipelineRunner.run_hyperopt] 搜索完成，共 {len(study.trials)} 次 trial，"
            f"最优值={study.best_value:.6f}，最优参数={best_params}"
        )
        results = objective.best_params_retrain(best_params)

        # ── 补充元信息（与原版 best_params_csv 列对齐） ────────────
        results.update({
            "data":             dataset_name,
            "model":            model_name,
            "featuring":        featurizer or "",
            "split_type":       split,
            "cv_mode":          cv_mode,
            "task":             task,
            "best_params":      str(best_params),
            "valid_loss_hopt":  study.best_value,
            "n_trials_done":    len(study.trials),
        })

        # ── 5. 保存逐样本预测 CSV ────────────────────────────────
        # best_params_retrain 已顺带计算了三个子集的预测，通过 _predictions 键返回
        if save_predictions and self.results_dir:
            preds_data = results.pop("_predictions", None)
            if preds_data:
                self._write_predictions_from_dict(
                    preds_data, model_name, featurizer or "", split, dataset_name
                )
        else:
            results.pop("_predictions", None)  # 清理内部键，不写入结果 CSV

        # ── 6. 写入汇总结果 CSV ───────────────────────────────────
        if self.result_csv:
            header = get_result_header(task, hyperopt=True)
            write_result(self.result_csv, results, header)

        logger.info(
            f"[PipelineRunner.run_hyperopt] 完成，"
            f"test={results.get('test_rmse', results.get('test_auc_roc', 'N/A'))}"
        )
        return results

    # ── 预测结果保存（内部辅助） ──────────────────────────────────────

    def _save_predictions(
        self,
        model: BaseModel,
        splits,
        task: str,
        model_name: str,
        featurizer: str,
        split: str,
        dataset_name: str,
    ) -> None:
        """
        对 train/valid/test 三个子集逐一预测并保存到 predictions/ CSV。

        文件名格式（对应原版）：
            {model_name}_{featurizer}_{split}_predictions_{dataset_name}.csv
        """
        pred_dir = self.results_dir / dataset_name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_csv = pred_dir / f"{model_name}_{featurizer}_{split}_predictions_{dataset_name}.csv"

        # 先删除旧文件（三个子集会追加写入，避免重复）
        if pred_csv.exists():
            pred_csv.unlink()

        start_idx = 0
        for subset_name, smiles_list, true_labels in [
            ("train", splits.train_smiles, splits.train_labels),
            ("valid", splits.val_smiles,   splits.val_labels),
            ("test",  splits.test_smiles,  splits.test_labels),
        ]:
            if not smiles_list:
                continue
            preds = model.predict(smiles_list)
            write_predictions(
                pred_csv, smiles_list,
                np.asarray(true_labels), preds,
                subset_name=subset_name,
                start_index=start_idx,
            )
            start_idx += len(smiles_list)

        logger.info(f"[PipelineRunner] 预测文件已保存: {pred_csv}")

    def _write_predictions_from_dict(
        self,
        preds_data: Dict,
        model_name: str,
        featurizer: str,
        split: str,
        dataset_name: str,
    ) -> None:
        """
        将 best_params_retrain 返回的 _predictions 字典写入预测 CSV 文件。

        preds_data 格式：
            { 'train': {'smiles': [...], 'true_label': array, 'predicted': array},
              'valid': {...},  'test': {...} }
        """
        pred_dir = self.results_dir / dataset_name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_csv = pred_dir / (
            f"{model_name}_{featurizer}_{split}_predictions_{dataset_name}.csv"
        )

        if pred_csv.exists():
            pred_csv.unlink()

        start_idx = 0
        for subset_name in ("train", "valid", "test"):
            subset = preds_data.get(subset_name)
            if not subset or not subset.get("smiles"):
                continue
            write_predictions(
                pred_csv,
                smiles_list=subset["smiles"],
                true_labels=np.asarray(subset["true_label"]),
                pred_labels=np.asarray(subset["predicted"]),
                subset_name=subset_name,
                start_index=start_idx,
            )
            start_idx += len(subset["smiles"])

        logger.info(f"[PipelineRunner] 预测文件已保存: {pred_csv}")

    # ── 数据划分缓存路径生成 ─────────────────────────────────────────

    def _split_cache_path(
        self,
        dataset_name: str,
        split: str,
        cv_mode: str,
    ) -> Optional[Path]:
        """
        生成划分缓存 pkl 路径（对应原版 split_indices_{split_type}_{cv_mode}.pkl）。
        若未配置 results_dir，则返回 None（不使用缓存）。
        """
        if not self.results_dir:
            return None
        cache_dir = self.results_dir / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"split_indices_{split}_{cv_mode}_{dataset_name}.pkl"

    # ── 模型评估（内部辅助） ─────────────────────────────────────────

    def _evaluate(self, model: BaseModel, splits, task: str) -> Dict:
        """
        在 train/valid/test 三个子集上评估模型，返回合并指标字典。

        使用 valid 前缀（与原版 cpu_ml_gnn/evaluator.py 一致）。
        """
        result = {}
        for subset_name, smiles_list, labels_arr in [
            ("train", splits.train_smiles, splits.train_labels),
            ("valid", splits.val_smiles,   splits.val_labels),    # valid（非 val）
            ("test",  splits.test_smiles,  splits.test_labels),
        ]:
            if not smiles_list:
                continue
            preds = model.predict(smiles_list)
            metrics = calc_metrics(
                np.asarray(labels_arr), preds, task, prefix=subset_name
            )
            result.update(metrics)
        return result
