"""
envmolbench.featurizer — 特征化层

提供工厂函数 get_featurizer() 和可用特征化器的注册表。

支持的特征化方法：
  - 'morgan'        : Morgan 二元指纹（ECFP）
  - 'morgan_count'  : Morgan 计数指纹
  - 'maccs'         : MACCS 167 位结构键
  - 'mordred'       : Mordred 全量分子描述符（需安装 mordred）
  - 'graph'         : 分子图（用于 GNN，需安装 torch-geometric）
  - 'image'         : 分子结构图片（用于 CNN）
  - 'smiles'        : 原始 SMILES 字符串（用于 ChemBERTa/Chemprop 等）
"""
from .base import BaseFeaturizer
from .fingerprint import MorganFeaturizer, MorganCountFeaturizer, MACCSFeaturizer
from .descriptor import MordredFeaturizer
from .graph import GraphFeaturizer
from .image import MolImageFeaturizer
from .smiles_feat import SmilesFeaturizer

# 注册表：方法名 → 特征化器类（不含参数）
_FEATURIZER_REGISTRY = {
    "morgan": MorganFeaturizer,
    "morgan_count": MorganCountFeaturizer,
    "maccs": MACCSFeaturizer,
    "mordred": MordredFeaturizer,
    "graph": GraphFeaturizer,
    "image": MolImageFeaturizer,
    "smiles": SmilesFeaturizer,
}


def list_featurizers() -> list:
    """返回所有已注册的特征化方法名称列表。"""
    return sorted(_FEATURIZER_REGISTRY.keys())


def get_featurizer(method: str, **kwargs) -> BaseFeaturizer:
    """
    工厂函数：按名称创建特征化器实例。

    Args:
        method: 特征化方法名称（见 list_featurizers()）。
        **kwargs: 传递给特征化器构造函数的参数，如：
                  get_featurizer('morgan', radius=2, n_bits=1024)

    Returns:
        对应的 BaseFeaturizer 子类实例。

    Raises:
        ValueError: method 不在注册表中。
    """
    cls = _FEATURIZER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"未知的特征化方法 '{method}'。\n"
            f"可用方法: {list_featurizers()}"
        )
    return cls(**kwargs)


__all__ = [
    "BaseFeaturizer",
    "MorganFeaturizer",
    "MorganCountFeaturizer",
    "MACCSFeaturizer",
    "MordredFeaturizer",
    "GraphFeaturizer",
    "MolImageFeaturizer",
    "SmilesFeaturizer",
    "get_featurizer",
    "list_featurizers",
]
