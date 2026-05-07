"""
数据划分器单元测试。

覆盖：
    - split_data 各划分方法的基本功能
    - SplitResult 属性访问
    - 划分比例正确性
    - 无数据泄漏（train/val/test 集合不重叠）
    - 随机划分的可复现性（相同 seed 结果一致）

运行方式::

    python -m pytest envmolbench/tests/test_splitter.py -v
"""
import numpy as np
import pytest

from envmolbench.data.splitter import split_data, SplitResult


# ── 测试数据 ─────────────────────────────────────────────────────────

_SMILES = [
    "CC", "CCO", "c1ccccc1", "CC(=O)O", "CN",
    "CCN", "CCCO", "c1ccc(N)cc1", "CC(=O)OC", "CCC",
    "CCCC", "CCCCO", "c1ccccc1O", "CC(N)=O", "CCNCC",
    "c1ccc(O)cc1", "CCC(=O)O", "CCc1ccccc1", "CCOCC", "CC(C)O",
]
_LABELS = list(range(len(_SMILES)))


# ── 测试 SplitResult 数据类 ───────────────────────────────────────────

class TestSplitResult:
    def test_attributes_exist(self):
        splits = split_data(_SMILES, _LABELS, method="random", seed=42)
        assert hasattr(splits, "train_smiles")
        assert hasattr(splits, "train_labels")
        assert hasattr(splits, "val_smiles")
        assert hasattr(splits, "val_labels")
        assert hasattr(splits, "test_smiles")
        assert hasattr(splits, "test_labels")

    def test_property_accessors(self):
        """train/val/test 属性应返回 (smiles, labels) 元组。"""
        splits = split_data(_SMILES, _LABELS, method="random", seed=42)
        assert isinstance(splits.train, tuple) and len(splits.train) == 2
        assert isinstance(splits.val, tuple) and len(splits.val) == 2
        assert isinstance(splits.test, tuple) and len(splits.test) == 2

    def test_lengths_consistent(self):
        """每个子集中 smiles 和 labels 长度应一致。"""
        splits = split_data(_SMILES, _LABELS, method="random", seed=42)
        assert len(splits.train_smiles) == len(splits.train_labels)
        assert len(splits.val_smiles) == len(splits.val_labels)
        assert len(splits.test_smiles) == len(splits.test_labels)


# ── 测试随机划分 ──────────────────────────────────────────────────────

class TestRandomSplit:
    def test_total_size(self):
        """train + val + test 样本数之和应等于原始数据集大小。"""
        splits = split_data(_SMILES, _LABELS, method="random",
                            test_size=0.1, val_size=0.1, seed=42)
        total = (len(splits.train_smiles) + len(splits.val_smiles)
                 + len(splits.test_smiles))
        assert total == len(_SMILES)

    def test_no_overlap(self):
        """三个子集之间不应有重叠 SMILES。"""
        splits = split_data(_SMILES, _LABELS, method="random",
                            test_size=0.1, val_size=0.1, seed=42)
        train_set = set(splits.train_smiles)
        val_set = set(splits.val_smiles)
        test_set = set(splits.test_smiles)
        assert len(train_set & val_set) == 0, "train 和 val 有重叠"
        assert len(train_set & test_set) == 0, "train 和 test 有重叠"
        assert len(val_set & test_set) == 0, "val 和 test 有重叠"

    def test_reproducibility(self):
        """相同 seed 应产生相同划分结果。"""
        splits1 = split_data(_SMILES, _LABELS, method="random", seed=42)
        splits2 = split_data(_SMILES, _LABELS, method="random", seed=42)
        assert splits1.train_smiles == splits2.train_smiles
        assert splits1.test_smiles == splits2.test_smiles

    def test_different_seeds(self):
        """不同 seed 应产生不同划分结果（概率上）。"""
        splits1 = split_data(_SMILES, _LABELS, method="random", seed=42)
        splits2 = split_data(_SMILES, _LABELS, method="random", seed=99)
        # 两次划分大概率不同（极小概率相同，可接受）
        assert splits1.train_smiles != splits2.train_smiles or \
               splits1.test_smiles != splits2.test_smiles

    def test_approximate_ratios(self):
        """划分比例应在合理范围内（±15% 容忍）。"""
        n = len(_SMILES)
        splits = split_data(_SMILES, _LABELS, method="random",
                            test_size=0.2, val_size=0.1, seed=42)
        test_ratio = len(splits.test_smiles) / n
        val_ratio = len(splits.val_smiles) / n
        assert 0.05 <= test_ratio <= 0.35, f"测试集比例 {test_ratio:.2f} 超出预期范围"
        assert 0.0 <= val_ratio <= 0.25, f"验证集比例 {val_ratio:.2f} 超出预期范围"


# ── 测试 Scaffold 划分 ───────────────────────────────────────────────

class TestScaffoldSplit:
    def test_total_size(self):
        splits = split_data(_SMILES, _LABELS, method="scaffold",
                            test_size=0.1, val_size=0.1, seed=42)
        total = (len(splits.train_smiles) + len(splits.val_smiles)
                 + len(splits.test_smiles))
        assert total == len(_SMILES)

    def test_no_overlap(self):
        splits = split_data(_SMILES, _LABELS, method="scaffold",
                            test_size=0.1, val_size=0.1, seed=42)
        train_set = set(splits.train_smiles)
        val_set = set(splits.val_smiles)
        test_set = set(splits.test_smiles)
        assert len(train_set & test_set) == 0, "scaffold split train 和 test 有重叠"
        assert len(val_set & test_set) == 0, "scaffold split val 和 test 有重叠"

    def test_train_not_empty(self):
        splits = split_data(_SMILES, _LABELS, method="scaffold",
                            test_size=0.1, val_size=0.1, seed=42)
        assert len(splits.train_smiles) > 0, "训练集不应为空"


# ── 测试不同划分方法的接口一致性 ─────────────────────────────────────

class TestSplitMethodConsistency:
    @pytest.mark.parametrize("method", ["random", "scaffold"])
    def test_returns_split_result(self, method):
        """所有划分方法应返回 SplitResult 对象。"""
        splits = split_data(_SMILES, _LABELS, method=method, seed=42)
        assert isinstance(splits, SplitResult)

    @pytest.mark.parametrize("method", ["random", "scaffold"])
    def test_smiles_preserved(self, method):
        """划分后的所有 SMILES 应与原始集合完全一致（无丢失/新增）。"""
        splits = split_data(_SMILES, _LABELS, method=method, seed=42)
        all_smiles = (list(splits.train_smiles)
                      + list(splits.val_smiles)
                      + list(splits.test_smiles))
        assert sorted(all_smiles) == sorted(_SMILES)

    def test_invalid_method_raises(self):
        """无效的划分方法应引发异常。"""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            split_data(_SMILES, _LABELS, method="nonexistent_method", seed=42)
