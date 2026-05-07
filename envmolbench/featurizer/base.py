"""
特征化器抽象基类。

所有特征化器均继承 BaseFeaturizer，
提供统一的 fit / transform / fit_transform 接口。
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseFeaturizer(ABC):
    """
    分子特征化器基类。

    子类需实现 transform() 方法，将 SMILES 列表转换为特征矩阵。
    fit() 方法对无状态特征化器默认为空操作（返回 self）。
    """

    def fit(self, smiles: List[str], labels: Optional[np.ndarray] = None) -> "BaseFeaturizer":
        """
        拟合特征化器（无状态特征化器默认空操作）。

        Args:
            smiles: 训练集 SMILES 列表。
            labels: 训练集标签（部分特征化器不需要）。

        Returns:
            self（支持链式调用）。
        """
        return self

    @abstractmethod
    def transform(self, smiles: List[str]) -> np.ndarray:
        """
        将 SMILES 列表转换为特征矩阵。

        Args:
            smiles: SMILES 字符串列表。

        Returns:
            形状为 (n_samples, n_features) 的 float32 数组。
            无法解析的 SMILES 对应行填充 NaN。
        """

    def fit_transform(
        self, smiles: List[str], labels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """拟合后立即转换（便捷方法）。"""
        return self.fit(smiles, labels).transform(smiles)

    @property
    def name(self) -> str:
        """特征化器的名称标识，供注册表使用。"""
        return self.__class__.__name__
