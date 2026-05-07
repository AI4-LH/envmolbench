"""
统一数据集加载模块。

支持两种加载方式：
  1. 按内置名称加载：load_dataset("esol")
  2. 按文件路径加载：load_dataset("path/to/my_data.csv")

自动识别：
  - SMILES 列（列名含 'smi' 的列）
  - 标签列（label / y / target / activity / value / class 等）
  - 任务类型（标签唯一值 ≤ 2 时判断为分类，否则为回归）
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .registry import DATASET_REGISTRY, list_datasets

logger = logging.getLogger(__name__)

# 自动识别标签列时检索的候选列名（按优先级排列）
_LABEL_CANDIDATES = ["label", "y", "target", "activity", "value", "class", "toxicity"]

# 内置数据集根目录，优先读取环境变量 ENVMOLBENCH_DATA_DIR，
# 否则回退到包同级的 datasets/ 目录（适合开发模式）。
# pip 安装后建议设置环境变量：export ENVMOLBENCH_DATA_DIR=/path/to/datasets
_DEFAULT_DATASETS_DIR = (
    Path(os.environ["ENVMOLBENCH_DATA_DIR"])
    if "ENVMOLBENCH_DATA_DIR" in os.environ
    else Path(__file__).parent.parent.parent / "datasets"
)


def _detect_smiles_col(df: pd.DataFrame) -> str:
    """自动检测 SMILES 列名：返回第一个列名含 'smi' 的列。"""
    for col in df.columns:
        if "smi" in col.lower():
            return col
    raise ValueError(
        f"无法自动识别 SMILES 列。请确保列名含 'smi'（如 'smiles', 'SMILES'）。"
        f"\n当前列名: {list(df.columns)}"
    )


def _detect_label_col(df: pd.DataFrame, smiles_col: str) -> str:
    """
    自动检测标签列名：
    1. 优先匹配 _LABEL_CANDIDATES 中的列名（不区分大小写）。
    2. 若无匹配，选取非 SMILES 的第一个数值列。
    """
    lower_cols = {col.lower(): col for col in df.columns}

    for candidate in _LABEL_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]

    # 回退：取第一个数值列（排除 SMILES 列）
    for col in df.columns:
        if col == smiles_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"未找到标准标签列，自动选择数值列 '{col}' 作为标签。")
            return col

    raise ValueError(
        f"无法自动识别标签列。请确保列名包含以下关键词之一: {_LABEL_CANDIDATES}。"
        f"\n当前列名: {list(df.columns)}"
    )


def _detect_task_type(labels: np.ndarray) -> str:
    """
    根据标签值自动判断任务类型：
    - 唯一值 ≤ 2 且只包含 0/1 → 分类
    - 否则 → 回归
    """
    unique_vals = np.unique(labels[~np.isnan(labels)])
    if len(unique_vals) <= 2 and set(unique_vals.tolist()).issubset({0.0, 1.0}):
        return "classification"
    return "regression"


def load_dataset(
    name_or_path: str,
    smiles_col: Optional[str] = None,
    label_col: Optional[str] = None,
    task_type: Optional[str] = None,
    datasets_dir: Optional[str] = None,
    drop_na: bool = True,
    clean: bool = False,
    remove_stereo: bool = False,
) -> Tuple[list, np.ndarray, str]:
    """
    加载分子数据集，返回 (smiles_list, labels, task_type)。

    Args:
        name_or_path:  内置数据集名称（如 'esol'）或 CSV 文件路径。
        smiles_col:    SMILES 列名；为 None 时自动检测。
        label_col:     标签列名；为 None 时自动检测。
        task_type:     任务类型 'regression' 或 'classification'；为 None 时自动判断。
        datasets_dir:  内置数据集根目录；为 None 时使用默认路径。
        drop_na:       是否删除 SMILES 或标签为空的行，默认 True。
        clean:         是否对 SMILES 进行标准化清洗（去盐、规范化、去重）。
                       对应原版 clean_data.py 的处理流程。
                       内置的 47 个数据集已预先清洗，通常无需设置为 True。
                       对自定义 CSV 建议设置为 True 以保证 SMILES 质量。
                       默认 False（保持原版 main_runner._load_data 行为，不额外清洗）。
        remove_stereo: clean=True 时是否移除立体化学信息，默认 False。

    Returns:
        smiles: SMILES 字符串列表。
        labels: 标签数组（float32）。
        task_type: 'regression' 或 'classification'。
    """
    # 解析文件路径
    csv_path = _resolve_path(name_or_path, datasets_dir)
    logger.info(f"加载数据集: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"原始行数: {len(df)}，列名: {list(df.columns)}")

    # 列名识别
    smi_col = smiles_col or _detect_smiles_col(df)
    lbl_col = label_col or _detect_label_col(df, smi_col)
    logger.info(f"SMILES 列: '{smi_col}'，标签列: '{lbl_col}'")

    # 删除空值行
    if drop_na:
        before = len(df)
        df = df.dropna(subset=[smi_col, lbl_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"删除含空值的行: {dropped} 行，剩余: {len(df)} 行")

    smiles: list = df[smi_col].tolist()
    labels: np.ndarray = df[lbl_col].values.astype(np.float32)

    # 可选：SMILES 标准化清洗（对应原版 clean_data.py 流程）
    if clean:
        logger.info("启用 SMILES 标准化清洗（去盐 + 规范化 + 去重）...")
        from .cleaner import standardize_smiles_series
        smiles, labels = standardize_smiles_series(
            smiles, labels, remove_stereo=remove_stereo
        )
        logger.info(f"清洗后剩余: {len(smiles)} 条样本")

    # 任务类型判断
    detected_task = task_type or _detect_task_type(labels)
    logger.info(f"任务类型: {detected_task}（共 {len(smiles)} 条样本）")

    return smiles, labels, detected_task


def _resolve_path(name_or_path: str, datasets_dir: Optional[str]) -> Path:
    """
    将名称或路径解析为实际 CSV 文件的 Path 对象。

    优先级：
    1. 若是已有文件路径，直接返回。
    2. 在注册表中查找内置数据集名称。
    3. 尝试在 datasets_dir 目录下直接查找同名文件。
    """
    # 情况1：直接路径
    direct_path = Path(name_or_path)
    if direct_path.exists():
        return direct_path

    # 情况2：注册表查找
    datasets_root = Path(datasets_dir) if datasets_dir else _DEFAULT_DATASETS_DIR

    if name_or_path in DATASET_REGISTRY:
        filename = DATASET_REGISTRY[name_or_path]
        path = datasets_root / filename
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Built-in dataset '{name_or_path}' not found at: {path}\n"
            f"Current datasets directory: {datasets_root}\n"
            f"Fix options:\n"
            f"  1. Set environment variable: ENVMOLBENCH_DATA_DIR=/path/to/datasets\n"
            f"  2. Pass datasets_dir= explicitly: load_dataset('{name_or_path}', datasets_dir='/path/to/datasets')"
        )

    # 情况3：尝试直接在目录下查找
    fallback = datasets_root / name_or_path
    if fallback.exists():
        return fallback

    # 均失败
    available = list_datasets()
    raise ValueError(
        f"未知数据集 '{name_or_path}'。\n"
        f"可用的内置数据集: {available}\n"
        f"或直接传入 CSV 文件路径。"
    )
