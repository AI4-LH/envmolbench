"""
数据划分模块。

对 cpu_ml_gnn/data_splitter.py 中的 DataSplitter 类进行薄封装，
暴露更简洁的 split_data() 函数，使 notebook 中的调用更简单。

所有底层划分逻辑保留在 DataSplitter 类中，不重复实现。
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd

# 从私有核心模块导入完整的 DataSplitter 类
from ._core_splitter import DataSplitter

logger = logging.getLogger(__name__)

# 支持的划分方法名称
SPLIT_METHODS = ["scaffold", "random", "time", "butina", "maxmin"]


@dataclass
class SplitResult:
    """
    封装数据划分结果，方便 notebook 中按名称访问各子集。

    属性：
        train / val / test: 每个子集的 (smiles列表, labels数组) 元组。
        indices: {'train': [...], 'val': [...], 'test': [...]} 原始索引。
    """
    train_smiles: List[str]
    train_labels: np.ndarray
    val_smiles: List[str]
    val_labels: np.ndarray
    test_smiles: List[str]
    test_labels: np.ndarray
    indices: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def train(self) -> Tuple[List[str], np.ndarray]:
        return self.train_smiles, self.train_labels

    @property
    def val(self) -> Tuple[List[str], np.ndarray]:
        return self.val_smiles, self.val_labels

    @property
    def test(self) -> Tuple[List[str], np.ndarray]:
        return self.test_smiles, self.test_labels


def split_data(
    smiles: List[str],
    labels: np.ndarray,
    method: str = "scaffold",
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    n_folds: int = 1,
    time_col: Optional[List] = None,
    cache_path: Optional[Union[str, Path]] = None,
) -> Union[SplitResult, List[SplitResult]]:
    """
    对分子数据集进行训练/验证/测试划分，或交叉验证划分。

    Args:
        smiles:     SMILES 字符串列表。
        labels:     标签数组。
        method:     划分方法（'scaffold' / 'random' / 'time' / 'butina' / 'maxmin'）。
        test_size:  测试集比例，默认 0.1。
        val_size:   验证集比例，默认 0.1（仅 n_folds=1 时使用）。
        seed:       随机种子，默认 42。
        n_folds:    折数，默认 1（单次划分）；>1 时返回 List[SplitResult]。
        time_col:   时间列（用于 'time' 划分方法），可传入时间值列表。
        cache_path: 索引缓存 pkl 路径；非 None 时自动读写缓存。

    Returns:
        n_folds=1 时返回单个 SplitResult；
        n_folds>1 时返回 List[SplitResult]，长度等于折数。
    """
    if method not in SPLIT_METHODS:
        raise ValueError(
            f"不支持的划分方法 '{method}'，可用方法: {SPLIT_METHODS}"
        )

    logger.info(
        f"执行数据划分：方法={method}，n_folds={n_folds}，样本数={len(smiles)}，seed={seed}"
    )

    # 构建 DataFrame 供 DataSplitter 使用
    df = pd.DataFrame({"smiles": smiles, "label": labels})
    if time_col is not None:
        df["time"] = time_col

    splitter = DataSplitter(
        dataset=df,
        smiles_col="smiles",
        label_col="label",
        time_col="time" if time_col is not None else None,
        cache_path=cache_path,
    )

    smiles_arr = list(smiles)

    if n_folds == 1:
        train_idx, val_idx, test_idx = _call_split_method(
            splitter, method, test_size, val_size, seed
        )
        logger.info(
            f"划分完成：训练集={len(train_idx)}，验证集={len(val_idx)}，测试集={len(test_idx)}"
        )
        return SplitResult(
            train_smiles=[smiles_arr[i] for i in train_idx],
            train_labels=labels[train_idx],
            val_smiles=[smiles_arr[i] for i in val_idx],
            val_labels=labels[val_idx],
            test_smiles=[smiles_arr[i] for i in test_idx],
            test_labels=labels[test_idx],
            indices={"train": train_idx, "val": val_idx, "test": test_idx},
        )

    # 交叉验证模式
    fold_indices = _call_cv_split_method(splitter, method, n_folds, test_size, seed)
    logger.info(f"交叉验证划分完成：{len(fold_indices)} 折")
    return [
        SplitResult(
            train_smiles=[smiles_arr[i] for i in tr],
            train_labels=labels[tr],
            val_smiles=[smiles_arr[i] for i in va],
            val_labels=labels[va],
            test_smiles=[smiles_arr[i] for i in te],
            test_labels=labels[te],
            indices={"train": tr, "val": va, "test": te},
        )
        for tr, va, te in fold_indices
    ]


def _call_split_method(
    splitter: DataSplitter,
    method: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """内部函数：根据方法名调用 DataSplitter 对应方法。"""
    if method == "scaffold":
        return splitter.get_scaffold_train_val_test_split(
            test_size=test_size, valid_size=val_size
        )
    elif method == "random":
        return splitter.get_random_train_val_test_split(
            test_size=test_size, valid_size=val_size, seed=seed
        )
    elif method == "time":
        return splitter.get_time_train_val_test_split(
            test_size=test_size, valid_size=val_size
        )
    elif method == "butina":
        return splitter.get_butina_train_val_test_split(
            test_size=test_size, valid_size=val_size
        )
    elif method == "maxmin":
        return splitter.get_maxmin_train_val_test_split(
            test_size=test_size, valid_size=val_size, seed=seed
        )
    else:
        raise ValueError(f"未知的划分方法: '{method}'")


def _call_cv_split_method(
    splitter: DataSplitter,
    method: str,
    n_folds: int,
    test_size: float,
    seed: int,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """内部函数：根据方法名调用 DataSplitter 对应的交叉验证方法。"""
    if method == "scaffold":
        return splitter.get_scaffold_cv_splits(n_splits=n_folds, test_size=test_size, seed=seed)
    elif method == "random":
        return splitter.get_random_cv_splits(n_splits=n_folds, test_size=test_size, seed=seed)
    elif method == "time":
        return splitter.get_time_cv_splits(n_splits=n_folds, test_size=test_size)
    elif method == "butina":
        return splitter.get_butina_cv_splits(n_splits=n_folds, test_size=test_size)
    elif method == "maxmin":
        return splitter.get_maxmin_cv_splits(n_splits=n_folds, test_size=test_size, seed=seed)
    else:
        raise ValueError(f"未知的划分方法: '{method}'")
