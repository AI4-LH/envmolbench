"""
特征化器单元测试。

覆盖：
    - get_featurizer 工厂函数
    - list_featurizers 返回完整列表
    - MorganFeaturizer：输出维度、数值类型
    - MACCSFeaturizer：固定 167 维输出
    - SmilesFeaturizer：直传 SMILES，形状一致
    - BaseFeaturizer 接口（fit/transform/fit_transform）

运行方式::

    python -m pytest envmolbench/tests/test_featurizer.py -v
"""
import numpy as np
import pytest

from envmolbench.featurizer import get_featurizer, list_featurizers
from envmolbench.featurizer.fingerprint import MorganFeaturizer, MACCSFeaturizer
from envmolbench.featurizer.smiles_feat import SmilesFeaturizer


# ── 测试数据 ─────────────────────────────────────────────────────────

_SMILES = [
    "CCO",          # 乙醇
    "c1ccccc1",     # 苯
    "CC(=O)O",      # 乙酸
    "CN",           # 甲胺
    "CC(=O)OC",     # 乙酸甲酯
]

_LABELS = [0.1, 1.2, 0.8, 0.3, 0.4]


# ── 测试 get_featurizer 和 list_featurizers ───────────────────────────

class TestFeaturizerRegistry:
    def test_list_featurizers_returns_list(self):
        result = list_featurizers()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_contains_morgan(self):
        result = list_featurizers()
        assert any("morgan" in f.lower() for f in result)

    def test_list_contains_maccs(self):
        result = list_featurizers()
        assert any("maccs" in f.lower() for f in result)

    @pytest.mark.parametrize("name", ["morgan", "maccs", "smiles"])
    def test_get_featurizer_valid(self, name):
        """有效名称应成功创建特征化器对象。"""
        feat = get_featurizer(name)
        assert feat is not None

    def test_get_featurizer_invalid_raises(self):
        """无效名称应引发 KeyError 或 ValueError。"""
        with pytest.raises((KeyError, ValueError)):
            get_featurizer("nonexistent_featurizer_xyz")


# ── 测试 MorganFeaturizer ─────────────────────────────────────────────

class TestMorganFeaturizer:
    def test_output_shape(self):
        """输出形状应为 (n_samples, n_bits)。"""
        n_bits = 1024
        feat = MorganFeaturizer(radius=2, n_bits=n_bits)
        X = feat.transform(_SMILES)
        assert X.shape == (len(_SMILES), n_bits)

    def test_default_nbits(self):
        """默认 n_bits=2048，输出列数应为 2048。"""
        feat = MorganFeaturizer()
        X = feat.transform(_SMILES)
        assert X.shape[1] == 2048

    def test_custom_radius(self):
        """不同 radius 应产生不同指纹（radius=1 vs radius=3）。"""
        feat1 = MorganFeaturizer(radius=1, n_bits=512)
        feat3 = MorganFeaturizer(radius=3, n_bits=512)
        X1 = feat1.transform(_SMILES)
        X3 = feat3.transform(_SMILES)
        # 两者应有差异
        assert not np.array_equal(X1, X3), "不同 radius 应产生不同指纹"

    def test_binary_values(self):
        """Morgan 二元指纹值应只包含 0 和 1。"""
        feat = MorganFeaturizer()
        X = feat.transform(_SMILES)
        unique_vals = set(X.flatten().tolist())
        assert unique_vals.issubset({0, 1, 0.0, 1.0})

    def test_dtype_numeric(self):
        """输出应为数值 numpy 数组。"""
        feat = MorganFeaturizer()
        X = feat.transform(_SMILES)
        assert isinstance(X, np.ndarray)
        assert np.issubdtype(X.dtype, np.number)

    def test_fit_transform_consistent(self):
        """fit_transform 结果应与单独 transform 一致。"""
        feat = MorganFeaturizer()
        X_ft = feat.fit_transform(_SMILES)
        X_t = feat.transform(_SMILES)
        np.testing.assert_array_equal(X_ft, X_t)

    def test_single_smiles(self):
        """单个 SMILES 也应正常返回 (1, n_bits) 形状。"""
        feat = MorganFeaturizer(n_bits=256)
        X = feat.transform(["CCO"])
        assert X.shape == (1, 256)


# ── 测试 MACCSFeaturizer ──────────────────────────────────────────────

class TestMACCSFeaturizer:
    def test_output_shape_167(self):
        """MACCS 键固定为 167 维。"""
        feat = MACCSFeaturizer()
        X = feat.transform(_SMILES)
        assert X.shape == (len(_SMILES), 167)

    def test_binary_values(self):
        feat = MACCSFeaturizer()
        X = feat.transform(_SMILES)
        unique_vals = set(X.flatten().tolist())
        assert unique_vals.issubset({0, 1, 0.0, 1.0})

    def test_dtype_numeric(self):
        feat = MACCSFeaturizer()
        X = feat.transform(_SMILES)
        assert isinstance(X, np.ndarray)


# ── 测试 SmilesFeaturizer ────────────────────────────────────────────

class TestSmilesFeaturizer:
    def test_returns_same_smiles(self):
        """SmilesFeaturizer 应直传 SMILES，不做任何变换。"""
        feat = SmilesFeaturizer()
        result = feat.transform(_SMILES)
        assert list(result) == list(_SMILES)

    def test_output_length(self):
        feat = SmilesFeaturizer()
        result = feat.transform(_SMILES)
        assert len(result) == len(_SMILES)


# ── 测试通过工厂函数创建后的一致性 ──────────────────────────────────────

class TestFeaturizerViaFactory:
    def test_morgan_via_factory(self):
        feat = get_featurizer("morgan", n_bits=512)
        X = feat.transform(_SMILES)
        assert X.shape == (len(_SMILES), 512)

    def test_maccs_via_factory(self):
        feat = get_featurizer("maccs")
        X = feat.transform(_SMILES)
        assert X.shape[1] == 167

    def test_smiles_via_factory(self):
        feat = get_featurizer("smiles")
        result = feat.transform(_SMILES)
        assert len(result) == len(_SMILES)
