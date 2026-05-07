"""
分子指纹特征化器。

迁移自 cpu_ml_gnn/feature_engineering.py，
重构为统一的 BaseFeaturizer 子类，并规范命名：
  - MorganBinaryFingerprint  → MorganFeaturizer（保留旧名作别名）
  - MorganCountFingerprint   → MorganCountFeaturizer
  - MACCSKeysFingerprint     → MACCSFeaturizer
"""
import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from .base import BaseFeaturizer

logger = logging.getLogger(__name__)


class MorganFeaturizer(BaseFeaturizer):
    """
    Morgan 二元指纹（ECFP）特征化器。

    Args:
        radius: 圆形邻域半径，默认 2（对应 ECFP4）。
        n_bits: 指纹位数，默认 2048。
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        if radius <= 0 or n_bits <= 0:
            raise ValueError("radius 和 n_bits 必须为正整数。")
        self.radius = int(radius)
        self.n_bits = int(n_bits)

    def transform(self, smiles: List[str]) -> np.ndarray:
        result = np.full((len(smiles), self.n_bits), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                result[i] = np.array(list(fp), dtype=np.float32)
            except Exception as e:
                logger.debug(f"SMILES '{smi}' 生成 Morgan 指纹失败: {e}")
        return result

    @property
    def name(self) -> str:
        return f"morgan_r{self.radius}_b{self.n_bits}"


# 保留旧名作别名，避免破坏已有调用
MorganBinaryFingerprint = MorganFeaturizer


class MorganCountFeaturizer(BaseFeaturizer):
    """
    Morgan 计数指纹特征化器（记录每个子结构出现次数，而非仅 0/1）。

    Args:
        radius: 圆形邻域半径，默认 2。
        n_bits: 指纹位数，默认 2048。
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        if radius <= 0 or n_bits <= 0:
            raise ValueError("radius 和 n_bits 必须为正整数。")
        self.radius = int(radius)
        self.n_bits = int(n_bits)

    def transform(self, smiles: List[str]) -> np.ndarray:
        result = np.full((len(smiles), self.n_bits), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.n_bits)
                vec = np.zeros(self.n_bits, dtype=np.float32)
                for idx, count in fp.GetNonzeroElements().items():
                    vec[idx] = count
                result[i] = vec
            except Exception as e:
                logger.debug(f"SMILES '{smi}' 生成 MorganCount 指纹失败: {e}")
        return result

    @property
    def name(self) -> str:
        return f"morgan_count_r{self.radius}_b{self.n_bits}"


# 保留旧名作别名
MorganCountFingerprint = MorganCountFeaturizer


class MACCSFeaturizer(BaseFeaturizer):
    """
    MACCS Keys 指纹特征化器（固定 167 位结构键）。
    """

    EXPECTED_LENGTH = 167

    def transform(self, smiles: List[str]) -> np.ndarray:
        result = np.full((len(smiles), self.EXPECTED_LENGTH), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                fp = list(MACCSkeys.GenMACCSKeys(mol))
                result[i] = np.array(fp, dtype=np.float32)
            except Exception as e:
                logger.debug(f"SMILES '{smi}' 生成 MACCS 指纹失败: {e}")
        return result

    @property
    def name(self) -> str:
        return "maccs"


# 保留旧名作别名
MACCSKeysFingerprint = MACCSFeaturizer
