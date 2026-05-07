"""
分子数据清洗与 SMILES 标准化模块。

迁移自 cpu_ml_gnn/clean_data.py，保留完整功能并补充中文注释。

主要功能：
    standardize_smiles()  —— 对单个 SMILES 进行去盐、标准化、多组分过滤
    clean_chemical_data() —— 批量清洗并保存为标准 CSV（label, smiles）
    standardize_smiles_series() —— 对 SMILES 列表就地标准化（用于 load_dataset 集成）

清洗步骤（与原版保持一致）：
    1. 解析 SMILES（RDKit MolFromSmiles）
    2. 去盐（RDKit SaltRemover，保留最大片段）
    3. （可选）移除立体化学信息
    4. 规范化为 canonical SMILES
    5. 过滤多组分 SMILES（含 '.'）
    6. 去重：先按 (canonical_smiles, label) 去重，再删除同 SMILES 不同 label 的冲突样本
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

logger = logging.getLogger(__name__)

# 禁用 RDKit 内部 C++ 日志，避免污染用户输出
RDLogger.DisableLog("rdApp.*")


# ── 单条 SMILES 标准化 ──────────────────────────────────────────────

def standardize_smiles(
    smiles: str,
    remover: SaltRemover.SaltRemover,
    remove_stereo: bool = False,
) -> Optional[str]:
    """
    对单个 SMILES 进行解析、去盐、（可选）去立体化学、规范化处理。

    对应原版 cpu_ml_gnn/clean_data.py 的 standardize_smiles()。

    Args:
        smiles:        输入的 SMILES 字符串。
        remover:       RDKit SaltRemover 实例（建议外部创建后复用，避免重复初始化）。
        remove_stereo: 是否移除立体化学信息，默认 False（保留立体化学）。

    Returns:
        成功返回规范化 SMILES 字符串；解析失败、多组分或空分子返回 None。
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"无法解析 SMILES: '{smiles}'")
            return None

        # 去盐：dontRemoveEverything=True 确保不会把所有原子都删除
        mol_desalted = remover.StripMol(mol, dontRemoveEverything=True)
        if mol_desalted is None:
            logger.warning(f"去盐后分子对象为空: '{smiles}'")
            return None

        # 可选：移除立体化学信息
        if remove_stereo:
            try:
                Chem.RemoveStereochemistry(mol_desalted)
            except Exception as stereo_err:
                logger.warning(f"移除 '{smiles}' 的立体化学信息时出错: {stereo_err}")

        # 生成规范化 SMILES
        canonical = Chem.MolToSmiles(mol_desalted, canonical=True)

        # 过滤多组分（去盐后仍含 '.'，说明有残留盐/溶剂）
        if "." in canonical:
            logger.warning(f"规范化后仍为多组分，已跳过: '{smiles}' → '{canonical}'")
            return None

        return canonical

    except Exception as exc:
        logger.error(f"处理 SMILES '{smiles}' 时发生异常: {exc}")
        return None


# ── 批量 SMILES 标准化（用于 load_dataset 内联调用） ─────────────────

def standardize_smiles_series(
    smiles_list: list,
    labels: np.ndarray,
    remove_stereo: bool = False,
) -> Tuple[list, np.ndarray]:
    """
    对 SMILES 列表批量标准化，同步过滤失败项并返回对齐的标签数组。

    用于 data/loader.py 的 load_dataset() 内联调用，无需写出临时文件。

    Args:
        smiles_list:   SMILES 字符串列表。
        labels:        对应标签的 numpy 数组（与 smiles_list 等长）。
        remove_stereo: 是否移除立体化学信息。

    Returns:
        (cleaned_smiles, cleaned_labels) — 已过滤无效项、已去重。
    """
    remover = SaltRemover.SaltRemover()
    canonical_list, label_list = [], []

    n_invalid = 0
    for smi, lbl in zip(smiles_list, labels):
        canon = standardize_smiles(smi, remover, remove_stereo=remove_stereo)
        if canon is not None:
            canonical_list.append(canon)
            label_list.append(float(lbl))
        else:
            n_invalid += 1

    if n_invalid > 0:
        logger.info(f"SMILES 标准化：{n_invalid} 个无效/多组分 SMILES 已移除，"
                    f"剩余 {len(canonical_list)}/{len(smiles_list)} 条。")

    # 去重 1：(canonical_smiles, label) 相同则保留一个
    df = pd.DataFrame({"smiles": canonical_list, "label": label_list})
    before_dedup = len(df)
    df.drop_duplicates(subset=["smiles", "label"], keep="first", inplace=True)
    logger.debug(f"(smiles, label) 去重：移除 {before_dedup - len(df)} 行")

    # 去重 2：同一 SMILES 对应不同标签（冲突）→ 全部删除
    before_conflict = len(df)
    df.drop_duplicates(subset=["smiles"], keep=False, inplace=True)
    n_conflict = before_conflict - len(df)
    if n_conflict > 0:
        logger.warning(f"检测到 {n_conflict} 行标签冲突（同一 SMILES 对应不同标签），已全部移除。")

    df.reset_index(drop=True, inplace=True)
    return df["smiles"].tolist(), df["label"].values.astype(np.float32)


# ── 文件级清洗（预处理工具，保存为新 CSV） ──────────────────────────

def clean_chemical_data(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    smiles_col: str = "smiles",
    label_col: str = "label",
    sheet_name: Union[int, str, None] = 0,
    remove_stereo: bool = False,
) -> Tuple[bool, int]:
    """
    加载、清洗并将化学数据保存为标准格式 CSV（列：label, smiles）。

    对应原版 cpu_ml_gnn/clean_data.py 的 clean_chemical_data()，功能完全对齐。

    清洗步骤：
        1. 加载 XLSX 或 CSV。
        2. 删除含空值的行。
        3. SMILES 标准化（去盐、规范化、可选去立体化学）。
        4. 过滤无效/多组分 SMILES。
        5. 去重：按 (canonical_smiles, label)，再删除标签冲突行。
        6. 保存为 label, smiles 格式的 CSV。

    Args:
        input_path:    输入文件路径（.xlsx 或 .csv）。
        output_path:   输出 CSV 文件路径。
        smiles_col:    SMILES 列名，默认 'smiles'。
        label_col:     标签列名，默认 'label'。
        sheet_name:    Excel 工作表名或索引（仅对 .xlsx 有效）。
        remove_stereo: 是否移除立体化学信息，默认 False。

    Returns:
        (success, final_row_count)
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    logger.info("─" * 50)
    logger.info(f"[cleaner] 开始清洗: {input_path}")
    logger.info(f"[cleaner] 输出路径: {output_path}")
    logger.info(f"[cleaner] 移除立体化学: {'是' if remove_stereo else '否'}")

    if not input_path.exists():
        logger.error(f"输入文件未找到: {input_path}")
        return False, 0

    # ── 1. 加载数据 ──────────────────────────────────────────────
    try:
        suffix = input_path.suffix.lower()
        if suffix == ".xlsx":
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            if isinstance(df, dict):
                logger.info("读取多个工作表，合并中...")
                df = pd.concat(df.values(), ignore_index=True)
        elif suffix == ".csv":
            df = pd.read_csv(input_path)
        else:
            logger.error(f"不支持的文件类型: '{suffix}'，仅支持 .xlsx 和 .csv。")
            return False, 0
        initial_rows = len(df)
        logger.info(f"加载成功，共 {initial_rows} 行。")
    except Exception as exc:
        logger.error(f"加载文件 '{input_path}' 时出错: {exc}")
        return False, 0

    # ── 2. 检查必要列 ─────────────────────────────────────────────
    if smiles_col not in df.columns or label_col not in df.columns:
        logger.error(f"缺少必需列 '{smiles_col}' 或 '{label_col}'，"
                     f"当前列: {list(df.columns)}")
        return False, 0

    # ── 3. 基础清洗：删除空值行 ────────────────────────────────────
    df.dropna(subset=[smiles_col, label_col], inplace=True)
    df = df[df[smiles_col].astype(str).str.strip() != ""]
    after_na = len(df)
    logger.info(f"删除空值/空白 SMILES 行后，剩余 {after_na} 行"
                f"（移除 {initial_rows - after_na} 行）。")
    if df.empty:
        logger.warning("基础清洗后数据为空，终止。")
        return True, 0

    # ── 4. SMILES 标准化 ─────────────────────────────────────────
    logger.info("开始 SMILES 标准化...")
    remover = SaltRemover.SaltRemover()
    df["_canonical_"] = df[smiles_col].apply(
        lambda s: standardize_smiles(s, remover, remove_stereo=remove_stereo)
    )

    # 过滤无效/多组分
    before_filter = len(df)
    df.dropna(subset=["_canonical_"], inplace=True)
    after_filter = len(df)
    logger.info(f"过滤无效/多组分后，剩余 {after_filter} 行"
                f"（移除 {before_filter - after_filter} 行）。")
    if df.empty:
        logger.warning("SMILES 标准化后数据为空，终止。")
        return True, 0

    # ── 5. 去重 ───────────────────────────────────────────────────
    # 去重 1：(canonical, label) 完全相同 → 保留一个
    before_d1 = len(df)
    df.drop_duplicates(subset=["_canonical_", label_col], keep="first", inplace=True)
    logger.info(f"(smiles, label) 去重后，剩余 {len(df)} 行"
                f"（移除 {before_d1 - len(df)} 行）。")

    # 去重 2：同一 canonical 对应不同标签（冲突）→ 全部删除
    before_d2 = len(df)
    df.drop_duplicates(subset=["_canonical_"], keep=False, inplace=True)
    n_conflict = before_d2 - len(df)
    if n_conflict > 0:
        logger.warning(f"删除标签冲突行: {n_conflict} 行（同一 SMILES 对应不同标签）。")

    # ── 6. 保存 ──────────────────────────────────────────────────
    df_out = df[[label_col, "_canonical_"]].rename(
        columns={"_canonical_": "smiles", label_col: "label"}
    ).reset_index(drop=True)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False, encoding="utf-8")
        final_rows = len(df_out)
        logger.info(f"[cleaner] 清洗完成，{final_rows} 条记录已保存至: {output_path}")
        logger.info("─" * 50)
        return True, final_rows
    except Exception as exc:
        logger.error(f"保存结果至 '{output_path}' 时出错: {exc}")
        return False, 0
