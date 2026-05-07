"""
SMILES 直传特征化器。

对于使用内置 Tokenizer 的模型（如 ChemBERTa），
无需额外特征化，直接将 SMILES 列表原样返回即可。
此特征化器作为占位符，使 API 调用保持一致。
"""
import logging
from typing import List

import numpy as np

from .base import BaseFeaturizer

logger = logging.getLogger(__name__)


class SmilesFeaturizer(BaseFeaturizer):
    """
    SMILES 字符串直传特征化器。

    transform() 返回 SMILES 字符串的 numpy 数组（dtype=object），
    供 ChemBERTa、Chemprop 等内部自带 tokenizer/图构建 的模型使用。
    """

    def transform(self, smiles: List[str]) -> np.ndarray:
        """直接返回 SMILES 字符串数组（不做任何转换）。"""
        return np.array(smiles, dtype=object)

    @property
    def name(self) -> str:
        return "smiles"
