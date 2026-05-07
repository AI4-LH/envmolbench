"""
模型抽象基类。

所有模型封装类继承 BaseModel，提供统一的
fit / predict / save / load 接口。
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    分子属性预测模型基类。

    子类需实现：
      - fit()    : 训练
      - predict(): 推理（返回连续值或概率）
    可选实现：
      - save()   : 保存权重
      - load()   : 加载权重
    """

    def __init__(self, task_type: str, config: Optional[Dict] = None):
        """
        Args:
            task_type: 'regression' 或 'classification'。
            config: 模型配置字典（从 YAML 合并而来）。
        """
        if task_type not in ("regression", "classification"):
            raise ValueError(f"task_type 必须为 'regression' 或 'classification'，得到: '{task_type}'")
        self.task_type = task_type
        self.config = config or {}
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """
        训练模型。

        Args:
            train_smiles: 训练集 SMILES 列表。
            train_labels: 训练集标签数组。
            val_smiles: 验证集 SMILES（用于早停）。
            val_labels: 验证集标签。

        Returns:
            self（支持链式调用）。
        """

    @abstractmethod
    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        生成预测值。

        Returns:
            回归任务：连续值数组（float32）。
            分类任务：正类概率数组（float32，值域 [0, 1]）。
        """

    def save(self, path: Union[str, Path]) -> None:
        """保存模型到指定路径（子类可覆盖）。"""
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 save() 方法。")

    def load(self, path: Union[str, Path]) -> "BaseModel":
        """从指定路径加载模型（子类可覆盖）。"""
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 load() 方法。")

    @property
    def name(self) -> str:
        """模型名称标识，供注册表使用。"""
        return self.__class__.__name__

    def _check_fitted(self) -> None:
        """检查模型是否已训练，未训练时抛出错误。"""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} 尚未训练，请先调用 fit()。")
