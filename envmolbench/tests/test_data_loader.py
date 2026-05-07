"""
数据加载器单元测试。

覆盖：
    - load_dataset 按名称加载内置数据集
    - load_dataset 按路径加载自定义 CSV
    - SMILES 列自动检测
    - 标签列自动检测
    - 任务类型自动检测（回归 vs 分类）
    - list_datasets 返回完整列表

运行方式::

    python -m pytest envmolbench/tests/test_data_loader.py -v
"""
import os
import tempfile

import numpy as np
import pytest

from envmolbench.data import load_dataset, list_datasets
from envmolbench.data.loader import _detect_smiles_col, _detect_label_col, _detect_task_type


# ── 测试数据（最小化 SMILES 集合，不依赖外部数据集文件）──────────────

# 简单有机分子 SMILES（全部合法）
_SMILES_LIST = [
    "CC",           # 乙烷
    "CCO",          # 乙醇
    "c1ccccc1",     # 苯
    "CC(=O)O",      # 乙酸
    "CN",           # 甲胺
    "CCN",          # 乙胺
    "CCCO",         # 正丙醇
    "c1ccc(N)cc1",  # 苯胺
    "CC(=O)OC",     # 乙酸甲酯
    "CCC",          # 丙烷
]

_REGRESSION_LABELS = [0.1, 0.5, 1.2, 0.8, 0.3, 0.6, 0.9, 1.1, 0.4, 0.7]
_CLASSIFICATION_LABELS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


# ── 辅助函数 ─────────────────────────────────────────────────────────

def _make_csv(smiles: list, labels: list, smiles_col: str = "smiles",
              label_col: str = "label") -> str:
    """创建临时 CSV 文件，返回路径。"""
    import pandas as pd
    df = pd.DataFrame({smiles_col: smiles, label_col: labels})
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ── 测试 list_datasets ───────────────────────────────────────────────

class TestListDatasets:
    def test_returns_list(self):
        result = list_datasets()
        assert isinstance(result, list), "list_datasets 应返回列表"

    def test_not_empty(self):
        result = list_datasets()
        assert len(result) > 0, "数据集列表不应为空"

    def test_contains_esol(self):
        """ESOL 是最常用的内置数据集，必须存在。"""
        result = list_datasets()
        assert any("esol" in ds.lower() for ds in result), \
            "内置数据集应包含 esol"


# ── 测试 _detect_smiles_col ──────────────────────────────────────────

class TestDetectSmilesCol:
    def test_standard_name(self):
        """标准列名 'smiles' 应被正确识别。"""
        import pandas as pd
        df = pd.DataFrame({"smiles": _SMILES_LIST[:3], "label": [0, 1, 0]})
        assert _detect_smiles_col(df) == "smiles"

    def test_uppercase_smiles(self):
        """大写 'SMILES' 也应被识别。"""
        import pandas as pd
        df = pd.DataFrame({"SMILES": _SMILES_LIST[:3], "label": [0, 1, 0]})
        assert _detect_smiles_col(df).upper() == "SMILES"

    def test_mol_col_name(self):
        """'mol' 列名应被识别为 SMILES 列。"""
        import pandas as pd
        df = pd.DataFrame({"mol": _SMILES_LIST[:3], "activity": [0.1, 0.2, 0.3]})
        detected = _detect_smiles_col(df)
        assert detected == "mol"


# ── 测试 _detect_task_type ───────────────────────────────────────────

class TestDetectTaskType:
    def test_binary_labels_is_classification(self):
        """0/1 二元标签应识别为分类任务。"""
        import numpy as np
        task = _detect_task_type(np.array(_CLASSIFICATION_LABELS))
        assert task == "classification"

    def test_continuous_labels_is_regression(self):
        """连续浮点标签应识别为回归任务。"""
        import numpy as np
        task = _detect_task_type(np.array(_REGRESSION_LABELS))
        assert task == "regression"

    def test_multiclass_is_classification(self):
        """多类别整数标签应识别为分类任务。"""
        import numpy as np
        task = _detect_task_type(np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
        assert task == "classification"


# ── 测试 load_dataset（CSV 文件路径） ─────────────────────────────────

class TestLoadDatasetFromCsv:
    def test_regression_csv(self):
        """回归 CSV 正常加载，返回 (smiles, labels, 'regression')。"""
        csv_path = _make_csv(_SMILES_LIST, _REGRESSION_LABELS)
        try:
            smiles, labels, task = load_dataset(csv_path)
            assert len(smiles) == len(_SMILES_LIST)
            assert len(labels) == len(_REGRESSION_LABELS)
            assert task == "regression"
        finally:
            os.unlink(csv_path)

    def test_classification_csv(self):
        """分类 CSV 正常加载，返回 (smiles, labels, 'classification')。"""
        csv_path = _make_csv(_SMILES_LIST, _CLASSIFICATION_LABELS)
        try:
            smiles, labels, task = load_dataset(csv_path)
            assert len(smiles) == len(_SMILES_LIST)
            assert task == "classification"
        finally:
            os.unlink(csv_path)

    def test_explicit_task_type_override(self):
        """显式指定 task_type 应覆盖自动检测。"""
        csv_path = _make_csv(_SMILES_LIST, _REGRESSION_LABELS)
        try:
            smiles, labels, task = load_dataset(csv_path, task_type="classification")
            assert task == "classification"
        finally:
            os.unlink(csv_path)

    def test_non_standard_column_names(self):
        """非标准列名（如 'molecule', 'value'）也应被正确识别。"""
        csv_path = _make_csv(
            _SMILES_LIST, _REGRESSION_LABELS,
            smiles_col="molecule", label_col="value"
        )
        try:
            smiles, labels, task = load_dataset(csv_path)
            assert len(smiles) == len(_SMILES_LIST)
        finally:
            os.unlink(csv_path)

    def test_smiles_are_strings(self):
        """返回的 SMILES 列表应全为字符串。"""
        csv_path = _make_csv(_SMILES_LIST, _REGRESSION_LABELS)
        try:
            smiles, labels, task = load_dataset(csv_path)
            assert all(isinstance(s, str) for s in smiles)
        finally:
            os.unlink(csv_path)

    def test_labels_are_numeric(self):
        """返回的标签应为数值类型。"""
        csv_path = _make_csv(_SMILES_LIST, _REGRESSION_LABELS)
        try:
            smiles, labels, task = load_dataset(csv_path)
            assert all(isinstance(v, (int, float, np.floating, np.integer)) for v in labels)
        finally:
            os.unlink(csv_path)

    def test_nonexistent_file_raises(self):
        """不存在的文件应引发异常。"""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_dataset("/nonexistent/path/to/data.csv")
