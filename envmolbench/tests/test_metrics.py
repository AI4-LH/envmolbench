"""
评估指标单元测试。

覆盖：
    - calc_regression_metrics：RMSE / MAE / R²
    - calc_classification_metrics：准确率 / AUC-ROC / 精确率 / 召回率 / F1
    - calc_metrics：统一入口，根据 task 类型分发
    - prefix 前缀机制（train_ / val_ / test_）
    - 边界情况：完美预测、全零预测

运行方式::

    python -m pytest envmolbench/tests/test_metrics.py -v
"""
import math

import numpy as np
import pytest

from envmolbench.common.metrics import (
    calc_regression_metrics,
    calc_classification_metrics,
    calc_metrics,
)


# ────────────────────────────────────────────────────────────────
# 回归指标测试
# ────────────────────────────────────────────────────────────────

class TestRegressionMetrics:
    """测试 calc_regression_metrics。"""

    def test_perfect_prediction(self):
        """完美预测时 RMSE=0, MAE=0, R²=1。"""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = calc_regression_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_known_rmse(self):
        """验证 RMSE 计算：误差向量 [1,1,1,1] → RMSE=1.0。"""
        y_true = [0.0, 0.0, 0.0, 0.0]
        y_pred = [1.0, 1.0, 1.0, 1.0]
        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(1.0)

    def test_known_mae(self):
        """验证 MAE 计算。"""
        y_true = [0.0, 2.0, 4.0]
        y_pred = [1.0, 1.0, 3.0]  # 绝对误差：1, 1, 1 → MAE=1.0
        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(1.0)

    def test_r2_range(self):
        """R² 应在 (-inf, 1] 范围内，一般合理预测 > 0。"""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics["r2"] <= 1.0
        assert metrics["r2"] > 0.9  # 该预测应有较高 R²

    def test_prefix(self):
        """前缀应正确附加到指标键名。"""
        y = [1.0, 2.0, 3.0]
        metrics = calc_regression_metrics(y, y, prefix="test")
        assert "test_rmse" in metrics
        assert "test_mae" in metrics
        assert "test_r2" in metrics
        # 无前缀键不应存在
        assert "rmse" not in metrics

    def test_no_prefix(self):
        """无前缀时键名应直接为 'rmse', 'mae', 'r2'。"""
        y = [1.0, 2.0, 3.0]
        metrics = calc_regression_metrics(y, y)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_numpy_arrays(self):
        """接受 numpy array 输入。"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        metrics = calc_regression_metrics(y_true, y_pred)
        assert "rmse" in metrics
        assert math.isfinite(metrics["rmse"])

    def test_single_sample(self):
        """单样本时 R² 可能为 NaN，但 RMSE/MAE 应有有限值。"""
        metrics = calc_regression_metrics([2.0], [2.0])
        assert math.isfinite(metrics["rmse"])
        assert math.isfinite(metrics["mae"])


# ────────────────────────────────────────────────────────────────
# 分类指标测试
# ────────────────────────────────────────────────────────────────

class TestClassificationMetrics:
    """测试 calc_classification_metrics。"""

    # 模拟完美二分类预测
    _y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    _y_pred_perfect = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    _y_pred_proba = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.05, 0.95]

    def test_perfect_auc(self):
        """完美预测的 AUC-ROC 应为 1.0。"""
        metrics = calc_classification_metrics(self._y_true, self._y_pred_perfect)
        assert metrics["auc_roc"] == pytest.approx(1.0)

    def test_perfect_accuracy(self):
        """完美预测的准确率应为 1.0。"""
        metrics = calc_classification_metrics(self._y_true, self._y_pred_perfect)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_auc_range(self):
        """AUC-ROC 应在 [0, 1] 范围内。"""
        metrics = calc_classification_metrics(self._y_true, self._y_pred_proba)
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_prefix(self):
        """前缀应正确附加。"""
        metrics = calc_classification_metrics(self._y_true, self._y_pred_proba, prefix="val")
        assert "val_auc_roc" in metrics
        assert "val_accuracy" in metrics
        assert "auc_roc" not in metrics

    def test_required_keys_present(self):
        """所有必需的指标键应存在。"""
        metrics = calc_classification_metrics(self._y_true, self._y_pred_proba)
        required = {"accuracy", "auc_roc", "recall", "precision", "f1"}
        assert required.issubset(set(metrics.keys()))

    def test_all_same_prediction(self):
        """当预测全为同一类别时，应能正常返回（不崩溃）。"""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0.5, 0.5, 0.5, 0.5, 0.5]  # 全部预测为中间值
        metrics = calc_classification_metrics(y_true, y_pred)
        assert "auc_roc" in metrics


# ────────────────────────────────────────────────────────────────
# 统一入口 calc_metrics 测试
# ────────────────────────────────────────────────────────────────

class TestCalcMetrics:
    """测试统一入口 calc_metrics。"""

    def test_regression_dispatch(self):
        """task='regression' 时应返回回归指标。"""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = calc_metrics(y, y, task="regression")
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "auc_roc" not in metrics

    def test_classification_dispatch(self):
        """task='classification' 时应返回分类指标。"""
        y_true = [0, 1, 0, 1]
        y_pred = [0.1, 0.9, 0.2, 0.8]
        metrics = calc_metrics(y_true, y_pred, task="classification")
        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert "rmse" not in metrics

    def test_prefix_forwarded(self):
        """prefix 参数应正确传递到下层函数。"""
        y = [1.0, 2.0, 3.0]
        metrics = calc_metrics(y, y, task="regression", prefix="train")
        assert "train_rmse" in metrics
        assert "rmse" not in metrics

    def test_invalid_task_raises(self):
        """无效任务类型应引发 ValueError。"""
        with pytest.raises(ValueError):
            calc_metrics([1, 2, 3], [1, 2, 3], task="invalid_task")

    def test_values_are_finite(self):
        """所有返回指标值应为有限浮点数。"""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.9, 4.1, 4.9]
        metrics = calc_metrics(y_true, y_pred, task="regression")
        for k, v in metrics.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"指标 '{k}' 的值不是有限数：{v}"
