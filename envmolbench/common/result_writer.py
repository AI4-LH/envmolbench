"""
结果写入与断点续跑模块。

提供：
    write_result()        —— 线程安全地追加指标行到结果 CSV
    write_predictions()   —— 保存每个分子的预测值 CSV（对齐原版 predictions/ 目录）
    is_experiment_done()  —— 检查实验是否已完成，支持断点续跑

CSV 列名规范（与原版 cpu_ml_gnn/main_runner.py 对齐）：
    指标前缀：
        train_*  —— 训练集指标
        valid_*  —— 验证集指标（原版使用 valid_，非 val_）
        test_*   —— 测试集指标
    分类指标：
        accuracy, auc_roc, recall, precision, f1_score  （原版使用 f1_score，非 f1）

预测文件格式（原版 predictions/{model}_{feat}_{split}_predictions_{data}.csv）：
    index          —— 样本在原始数据集中的位置（整型，全局索引）
    smiles         —— SMILES 字符串
    set            —— 数据集划分标签（'train' / 'valid' / 'test'）
    true_label     —— 真实标签
    predicted_label —— 模型预测值（回归）或正类概率（分类）
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from filelock import FileLock

logger = logging.getLogger(__name__)


# ── 指标列名常量（与原版对齐） ────────────────────────────────────────

RESULT_HEADER_REGRESSION = [
    "data", "model", "featuring", "split_type", "task",
    "train_rmse", "train_mae", "train_r2",
    "valid_rmse", "valid_mae", "valid_r2",
    "test_rmse",  "test_mae",  "test_r2",
]

RESULT_HEADER_REGRESSION_HYPEROPT = [
    "data", "model", "featuring", "split_type", "cv_mode", "task",
    "best_params", "valid_loss_hopt",
    "train_rmse", "train_mae", "train_r2",
    "valid_rmse", "valid_mae", "valid_r2",
    "test_rmse",  "test_mae",  "test_r2",
    "n_trials_done",
]

RESULT_HEADER_CLASSIFICATION = [
    "data", "model", "featuring", "split_type", "task",
    "train_accuracy", "train_auc_roc", "train_recall", "train_precision", "train_f1_score",
    "valid_accuracy", "valid_auc_roc", "valid_recall", "valid_precision", "valid_f1_score",
    "test_accuracy",  "test_auc_roc",  "test_recall",  "test_precision",  "test_f1_score",
]

RESULT_HEADER_CLASSIFICATION_HYPEROPT = [
    "data", "model", "featuring", "split_type", "cv_mode", "task",
    "best_params", "valid_loss_hopt",
    "train_accuracy", "train_auc_roc", "train_recall", "train_precision", "train_f1_score",
    "valid_accuracy", "valid_auc_roc", "valid_recall", "valid_precision", "valid_f1_score",
    "test_accuracy",  "test_auc_roc",  "test_recall",  "test_precision",  "test_f1_score",
    "n_trials_done",
]

# 超参数搜索迭代日志的列名（对应原版 hyperopt_iterations_csv）
HYPEROPT_ITER_HEADER = [
    "data_name", "model_name", "featuring", "v_loss", "params", "ITERATION", "status",
]

# 预测文件的列名（对应原版 predictions/ 目录下的 CSV）
PREDICTION_HEADER = ["index", "smiles", "set", "true_label", "predicted_label"]


def get_result_header(task_type: str, hyperopt: bool = False) -> List[str]:
    """
    根据任务类型和是否为超参数搜索结果，返回对应的标准表头列表。

    Args:
        task_type: 'regression' 或 'classification'。
        hyperopt:  True 时返回包含 featuring/cv_mode/best_params 的扩展表头。
    """
    if task_type == "regression":
        return RESULT_HEADER_REGRESSION_HYPEROPT if hyperopt else RESULT_HEADER_REGRESSION
    elif task_type == "classification":
        return RESULT_HEADER_CLASSIFICATION_HYPEROPT if hyperopt else RESULT_HEADER_CLASSIFICATION
    else:
        raise ValueError(f"未知任务类型: '{task_type}'")


# ── 结果行写入 ─────────────────────────────────────────────────────

def write_result(
    result_csv: Union[str, Path],
    row: Dict,
    header: Optional[List[str]] = None,
) -> None:
    """
    线程安全地将一行结果追加到 CSV 文件。

    若文件不存在或为空，自动写入表头。
    使用 filelock 防止多进程并发写冲突。

    Args:
        result_csv: 目标 CSV 文件路径。
        row:        要写入的数据字典，键为列名。
        header:     列名列表；为 None 时使用 row.keys() 的顺序。
    """
    result_path = Path(result_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if header is None:
        header = list(row.keys())

    lock_path = result_path.with_suffix(result_path.suffix + ".lock")

    try:
        with FileLock(str(lock_path), timeout=10):
            file_exists = result_path.exists() and result_path.stat().st_size > 0
            with open(result_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
    except Exception as exc:
        logger.error(f"写入结果到 {result_path} 失败: {exc}", exc_info=True)


# ── 超参数迭代日志写入 ──────────────────────────────────────────────

def write_hyperopt_iter(
    log_csv: Union[str, Path],
    data_name: str,
    model_name: str,
    featuring: str,
    v_loss: float,
    params: dict,
    iteration: int,
    status: str = "ok",
) -> None:
    """
    将单次 hyperopt trial 的结果追加到迭代日志 CSV。

    对应原版 hyperopt_iterations_{iter}.csv。

    Args:
        log_csv:    迭代日志 CSV 路径。
        data_name:  数据集名称。
        model_name: 模型名称。
        featuring:  特征化器名称。
        v_loss:     验证集损失/分数。
        params:     本次 trial 的参数字典（字符串化存储）。
        iteration:  全局迭代计数器（Optuna 的 trial.number）。
        status:     'ok' 或错误描述字符串。
    """
    row = {
        "data_name":  data_name,
        "model_name": model_name,
        "featuring":  featuring,
        "v_loss":     v_loss,
        "params":     str(params),
        "ITERATION":  iteration,
        "status":     status,
    }
    write_result(log_csv, row, header=HYPEROPT_ITER_HEADER)


# ── 每分子预测结果写入 ──────────────────────────────────────────────

def write_predictions(
    pred_csv: Union[str, Path],
    smiles_list: List[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    subset_name: str,
    start_index: int = 0,
) -> None:
    """
    将单个子集（train/valid/test）的逐样本预测写入 CSV。

    多次调用（一次 train、一次 valid、一次 test）时会追加到同一文件。
    第一次调用时自动写入表头。

    对应原版 predictions/{model}_{feat}_{split}_predictions_{data}.csv。

    列格式：
        index           —— 样本在整个数据集（train+valid+test）中的顺序编号
        smiles          —— SMILES 字符串
        set             —— 'train' / 'valid' / 'test'
        true_label      —— 真实标签
        predicted_label —— 预测值（回归）或正类概率（分类）

    Args:
        pred_csv:     输出 CSV 文件路径（会追加，不覆盖）。
        smiles_list:  该子集的 SMILES 列表。
        true_labels:  真实标签数组。
        pred_labels:  预测值数组。
        subset_name:  'train' / 'valid' / 'test'。
        start_index:  该子集起始全局编号（train 从 0 开始，valid 从 len(train) 开始，etc.）。
    """
    pred_path = Path(pred_csv)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (smi, y_true, y_pred) in enumerate(zip(smiles_list, true_labels, pred_labels)):
        rows.append({
            "index":           start_index + i,
            "smiles":          smi,
            "set":             subset_name,
            "true_label":      float(y_true),
            "predicted_label": float(y_pred) if not np.isnan(float(y_pred)) else "",
        })

    if not rows:
        return

    lock_path = pred_path.with_suffix(pred_path.suffix + ".lock")
    try:
        with FileLock(str(lock_path), timeout=10):
            file_exists = pred_path.exists() and pred_path.stat().st_size > 0
            with open(pred_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PREDICTION_HEADER, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
    except Exception as exc:
        logger.error(f"写入预测文件 {pred_path} 失败: {exc}", exc_info=True)


# ── 断点续跑检查 ────────────────────────────────────────────────────

def is_experiment_done(
    result_csv: Union[str, Path],
    model_name: str,
    dataset_name: str,
    split_method: str,
    featuring: Optional[str] = None,
    cv_mode: Optional[str] = None,
) -> bool:
    """
    检查指定实验组合是否已存在于结果 CSV 中（用于断点续跑）。

    支持两种列名风格：
        - 新版列名：model / dataset / split
        - 原版列名：model / data  / split_type（自动兼容）

    Args:
        result_csv:   汇总结果 CSV 文件路径。
        model_name:   模型名称。
        dataset_name: 数据集名称。
        split_method: 数据划分方法。
        featuring:    特征化器名称（可选，仅超参数搜索结果使用）。
        cv_mode:      'single' 或 'cv5'（可选，仅超参数搜索结果使用）。

    Returns:
        True 表示已完成，可跳过；False 表示需要运行。
    """
    result_path = Path(result_csv)
    if not result_path.exists() or result_path.stat().st_size == 0:
        return False

    try:
        df = pd.read_csv(result_path)

        # 自动兼容新旧列名
        model_col   = "model"
        dataset_col = "dataset" if "dataset" in df.columns else "data"
        split_col   = "split"   if "split"   in df.columns else "split_type"

        mask = (
            (df[model_col]   == model_name)
            & (df[dataset_col] == dataset_name)
            & (df[split_col]   == split_method)
        )
        if featuring and "featuring" in df.columns:
            mask &= (df["featuring"] == featuring)
        if cv_mode and "cv_mode" in df.columns:
            mask &= (df["cv_mode"] == cv_mode)

        return not df[mask].empty

    except Exception as exc:
        logger.warning(f"读取结果文件 {result_path} 时出错: {exc}，将重新运行。")
        return False
