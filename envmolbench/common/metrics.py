"""
统一指标计算模块。

合并了 cpu_ml_gnn/evaluator.py 与各训练 notebook 中散落的
calculate_regression_metrics / calculate_classification_metrics 函数，
并统一命名为 calc_regression_metrics / calc_classification_metrics。
"""
import numpy as np
import logging
from typing import Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def calc_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    计算回归任务的评估指标：RMSE、MAE、R²。

    Args:
        y_true: 真实标签数组（一维）。
        y_pred: 预测值数组（一维）。
        prefix: 指标键名前缀，如 'train'、'val'、'test'。
                若非空，键名格式为 '{prefix}_rmse'；否则为 'rmse'。

    Returns:
        包含 rmse、mae、r2 三个键的字典（值可能为 np.nan）。
    """
    key = (lambda k: f"{prefix}_{k}" if prefix else k)

    # 输入校验：空数组或形状不匹配时返回 NaN
    if (
        y_true is None
        or y_pred is None
        or y_true.shape != y_pred.shape
        or y_true.ndim != 1
        or len(y_true) == 0
    ):
        return {key("rmse"): np.nan, key("mae"): np.nan, key("r2"): np.nan}

    try:
        return {
            key("rmse"): float(root_mean_squared_error(y_true, y_pred)),
            key("mae"): float(mean_absolute_error(y_true, y_pred)),
            key("r2"): float(r2_score(y_true, y_pred)),
        }
    except Exception as e:
        logger.error(f"计算回归指标时出错: {e}")
        return {key("rmse"): np.nan, key("mae"): np.nan, key("r2"): np.nan}


def calc_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    计算分类任务的评估指标：AUC-ROC、准确率、召回率、精确率、F1。

    Args:
        y_true: 真实二分类标签数组（0/1）。
        y_pred_proba: 正类预测概率数组（一维，值域 [0, 1]）。
        prefix: 指标键名前缀。

    Returns:
        包含 accuracy、auc_roc、recall、precision、f1 五个键的字典。
    """
    key = (lambda k: f"{prefix}_{k}" if prefix else k)

    nan_result = {
        key("accuracy"): np.nan,
        key("auc_roc"): np.nan,
        key("recall"): np.nan,
        key("precision"): np.nan,
        key("f1_score"): np.nan,    # 对应原版 evaluator.py 的 {prefix}_f1_score
    }

    if (
        y_true is None
        or y_pred_proba is None
        or len(y_true) != len(y_pred_proba)
        or len(y_true) == 0
    ):
        return nan_result

    try:
        y_true_int = y_true.astype(np.int32)
        y_pred_class = (y_pred_proba >= 0.5).astype(np.int32)

        # 若只有一个类别，AUC 无意义，返回 0.5
        auc = (
            float(roc_auc_score(y_true_int, y_pred_proba))
            if len(np.unique(y_true_int)) > 1
            else 0.5
        )

        return {
            key("accuracy"):  float(accuracy_score(y_true_int, y_pred_class)),
            key("auc_roc"):   auc,
            key("recall"):    float(recall_score(y_true_int, y_pred_class, zero_division=0)),
            key("precision"): float(precision_score(y_true_int, y_pred_class, zero_division=0)),
            key("f1_score"):  float(f1_score(y_true_int, y_pred_class, zero_division=0)),
        }
    except Exception as e:
        logger.error(f"计算分类指标时出错 (prefix='{prefix}'): {e}")
        return nan_result


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    prefix: str = "",
) -> Dict[str, float]:
    """
    根据任务类型自动选择并计算指标，作为统一入口。

    Args:
        y_true: 真实标签。
        y_pred: 预测值（回归）或正类概率（分类）。
        task_type: 'regression' 或 'classification'。
        prefix: 键名前缀。

    Returns:
        指标字典。
    """
    if task_type == "regression":
        return calc_regression_metrics(y_true, y_pred, prefix)
    elif task_type == "classification":
        return calc_classification_metrics(y_true, y_pred, prefix)
    else:
        raise ValueError(f"未知任务类型: '{task_type}'，应为 'regression' 或 'classification'。")
