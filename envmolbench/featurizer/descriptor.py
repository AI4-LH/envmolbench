"""
Mordred 分子描述符特征化器 + 特征筛选器。

迁移自 cpu_ml_gnn/feature_engineering.py 中的：
    - MolecularDescriptorsCalculator  → MordredFeaturizer
    - process_descriptors_numpy_modified → DescriptorSelector

⚠️ 依赖安装：
    pip install mordred-community
    （原始 mordred 包已停止维护，请使用社区维护版 mordred-community，
    但导入方式相同：from mordred import Calculator, descriptors）

特征筛选流程（DescriptorSelector.fit_transform / fit + transform）：
    1. 移除含 NaN / Inf 的列
    2. 低方差过滤（默认 threshold=0.0，移除零方差列）
    3. 常数列移除（std < 1e-10）
    4. 高相关性过滤（threshold=0.8，保留方差较大的特征）
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from scipy import stats

from .base import BaseFeaturizer

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# MordredFeaturizer
# ────────────────────────────────────────────────────────────────

class MordredFeaturizer(BaseFeaturizer):
    """
    使用 mordred-community 库计算全量 2D 分子描述符。

    安装依赖::

        pip install mordred-community

    Args:
        ignore_3d: 是否忽略 3D 描述符，默认 True（无需 3D 坐标）。

    示例::

        from envmolbench.featurizer.descriptor import MordredFeaturizer, DescriptorSelector

        feat = MordredFeaturizer()
        X_raw = feat.transform(train_smiles)      # shape (n, ~1800)

        selector = DescriptorSelector()
        X_train = selector.fit_transform(X_raw)   # 筛选后的训练集特征
        X_test  = selector.transform(feat.transform(test_smiles))
    """

    def __init__(self, ignore_3d: bool = True):
        self.ignore_3d = ignore_3d
        self._calculator = None
        self._desc_names: Optional[List[str]] = None

    def _ensure_calculator(self) -> None:
        """延迟初始化 mordred 计算器，避免 import 时崩溃。"""
        if self._calculator is not None:
            return
        try:
            # mordred-community 安装后，导入方式与原版相同
            from mordred import Calculator, descriptors as mordred_descs
            self._calculator = Calculator(mordred_descs, ignore_3D=self.ignore_3d)
            self._desc_names = [str(d) for d in self._calculator.descriptors]
            logger.info(
                f"Mordred 计算器初始化完成，共 {len(self._desc_names)} 个描述符。"
                f"（使用 mordred-community）"
            )
        except ImportError:
            raise ImportError(
                "使用 MordredFeaturizer 需要安装 mordred-community：\n"
                "    pip install mordred-community\n"
                "注意：原始 mordred 包已停止维护，请使用社区维护版。"
            )

    @property
    def descriptor_names(self) -> List[str]:
        """描述符名称列表（延迟初始化后可用）。"""
        self._ensure_calculator()
        return list(self._desc_names)

    def transform(self, smiles: List[str]) -> np.ndarray:
        """
        将 SMILES 列表转换为原始描述符矩阵（含 NaN）。

        注意：此方法返回**未筛选**的原始矩阵。
        使用 DescriptorSelector.fit_transform / transform 进行特征筛选。
        """
        self._ensure_calculator()
        n_desc = len(self._desc_names)
        result = np.full((len(smiles), n_desc), np.nan, dtype=np.float32)

        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mordred_result = self._calculator(mol)
                values = []
                for val in mordred_result:
                    # mordred 返回 Error 对象或数值
                    if hasattr(val, "__float__"):
                        try:
                            v = float(val)
                            values.append(v if np.isfinite(v) else np.nan)
                        except (TypeError, ValueError):
                            values.append(np.nan)
                    else:
                        values.append(np.nan)
                result[i] = np.array(values, dtype=np.float32)
            except Exception as e:
                logger.debug(f"SMILES '{smi}' 计算 Mordred 描述符失败: {e}")

        return result

    @property
    def name(self) -> str:
        return "mordred"


# ────────────────────────────────────────────────────────────────
# DescriptorSelector
# ────────────────────────────────────────────────────────────────

class DescriptorSelector:
    """
    分子描述符特征筛选器。

    迁移自 cpu_ml_gnn/feature_engineering.py 的
    process_descriptors_numpy_modified 函数，封装为 fit/transform 接口，
    确保只在训练数据上拟合筛选规则，避免验证/测试集数据泄漏。

    筛选步骤（按顺序）：
        1. 移除含 NaN / Inf 的列
        2. 低方差过滤（variance_threshold，默认 0.0 即移除零方差列）
        3. 常数列移除（std < 1e-10）
        4. 高相关性过滤（correlation_threshold=0.8）：
           成对比较时保留方差较大的特征，移除方差较小的那个

    属性：
        kept_indices_ (List[int]): fit 后保留的列索引（相对于原始描述符维度）。
        n_features_in_ (int): fit 时的原始特征数。
        n_features_out_ (int): 筛选后保留的特征数。

    示例::

        selector = DescriptorSelector(correlation_threshold=0.8)
        X_train_sel = selector.fit_transform(X_train_raw)   # 仅在训练集上 fit
        X_val_sel   = selector.transform(X_val_raw)         # 用相同列索引处理验证集
    """

    def __init__(
        self,
        correlation_threshold: float = 0.8,
        variance_threshold: float = 0.0,
    ):
        """
        Args:
            correlation_threshold: 相关系数上限，超过此值的特征对中保留方差大的。
                                   默认 0.8，与原版一致。
            variance_threshold:    方差下限，低于此值的列被移除。
                                   默认 0.0 即移除零方差列。
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.kept_indices_: Optional[List[int]] = None
        self.n_features_in_: Optional[int] = None
        self.n_features_out_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.kept_indices_ is not None

    def fit(self, X: np.ndarray) -> "DescriptorSelector":
        """
        在训练数据上拟合筛选规则，记录要保留的列索引。

        Args:
            X: 原始描述符矩阵，shape (n_samples, n_features)。

        Returns:
            self（支持链式调用）。
        """
        if X is None or X.size == 0:
            raise ValueError("输入描述符矩阵为空，无法 fit。")

        self.n_features_in_ = X.shape[1]
        # 用 absolute_indices 追踪每步筛选后保留列在原始矩阵中的位置
        absolute_indices = np.arange(X.shape[1])
        current_X = X.copy()

        logger.info(f"[DescriptorSelector] 开始特征筛选，原始特征数: {X.shape[1]}")

        # ── 步骤 1：移除含 NaN / Inf 的列 ────────────────────────
        invalid_mask = np.isnan(current_X).any(axis=0) | np.isinf(current_X).any(axis=0)
        keep = np.where(~invalid_mask)[0]
        removed = len(invalid_mask) - len(keep)
        if removed > 0:
            logger.info(f"  移除 {removed} 个含 NaN/Inf 的列，剩余: {len(keep)}")
        current_X = current_X[:, keep]
        absolute_indices = absolute_indices[keep]

        if current_X.shape[1] == 0:
            raise ValueError("步骤1（NaN/Inf过滤）后无剩余特征。")

        # ── 步骤 2：低方差过滤 ────────────────────────────────────
        variances = np.var(current_X, axis=0)
        keep = np.where(variances > self.variance_threshold)[0]
        removed = current_X.shape[1] - len(keep)
        if removed > 0:
            logger.info(
                f"  移除 {removed} 个低方差（≤{self.variance_threshold}）列，剩余: {len(keep)}"
            )
        current_X = current_X[:, keep]
        absolute_indices = absolute_indices[keep]

        if current_X.shape[1] == 0:
            raise ValueError("步骤2（低方差过滤）后无剩余特征。")

        # ── 步骤 3：常数列移除（std < 1e-10，比方差更严格的兜底） ──
        std_dev = np.std(current_X, axis=0)
        keep = np.where(std_dev >= 1e-10)[0]
        removed = current_X.shape[1] - len(keep)
        if removed > 0:
            logger.info(f"  移除 {removed} 个常数列（std<1e-10），剩余: {len(keep)}")
        current_X = current_X[:, keep]
        absolute_indices = absolute_indices[keep]

        if current_X.shape[1] < 2:
            # 少于 2 列时无法计算相关性，直接跳过
            self.kept_indices_ = absolute_indices.tolist()
            self.n_features_out_ = len(self.kept_indices_)
            logger.info(
                f"[DescriptorSelector] 特征数 < 2，跳过相关性过滤。"
                f"最终保留: {self.n_features_out_} 个特征"
            )
            return self

        # ── 步骤 4：高相关性过滤 ─────────────────────────────────
        # 使用 pearsonr 安全计算相关矩阵（处理常数列的 NaN）
        n_feat = current_X.shape[1]
        corr_matrix = np.ones((n_feat, n_feat), dtype=np.float32)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                try:
                    corr, _ = stats.pearsonr(current_X[:, i], current_X[:, j])
                    corr = 0.0 if np.isnan(corr) else corr
                except Exception:
                    corr = 0.0
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        corr_abs = np.abs(corr_matrix)
        variances = np.var(current_X, axis=0)

        to_drop = set()
        for i in range(n_feat):
            if i in to_drop:
                continue
            for j in range(i + 1, n_feat):
                if j in to_drop:
                    continue
                if corr_abs[i, j] > self.correlation_threshold:
                    # 移除方差较小的特征
                    if variances[i] >= variances[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)
                        break  # i 已被标记，跳出内层

        keep = [idx for idx in range(n_feat) if idx not in to_drop]
        removed = n_feat - len(keep)
        if removed > 0:
            logger.info(
                f"  移除 {removed} 个高相关（>{self.correlation_threshold}）列，剩余: {len(keep)}"
            )
        current_X = current_X[:, keep]
        absolute_indices = absolute_indices[keep]

        self.kept_indices_ = absolute_indices.tolist()
        self.n_features_out_ = len(self.kept_indices_)
        logger.info(
            f"[DescriptorSelector] 筛选完成："
            f"{self.n_features_in_} → {self.n_features_out_} 个特征"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用 fit 时确定的列索引，从特征矩阵中提取保留列。

        Args:
            X: 描述符矩阵，列数须与 fit 时一致。

        Returns:
            筛选后的特征矩阵。
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 或 fit_transform()。")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"特征维度不匹配：fit 时 {self.n_features_in_}，"
                f"transform 时 {X.shape[1]}。"
            )
        return X[:, self.kept_indices_].astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在同一数据上执行 fit 后立即 transform（仅供训练集使用）。"""
        return self.fit(X).transform(X)

    def verify_correlation(self, X: np.ndarray) -> bool:
        """
        验证筛选后的特征矩阵中最大相关系数是否低于阈值。
        通常在 fit_transform 后调用，用于日志确认。
        """
        if X.shape[1] < 2:
            return True
        try:
            corr = np.corrcoef(X, rowvar=False)
            np.nan_to_num(corr, copy=False, nan=0.0)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            max_corr = float(np.max(np.abs(corr[mask])))
            if max_corr >= self.correlation_threshold:
                logger.warning(
                    f"[DescriptorSelector] 验证：最大相关 {max_corr:.4f} "
                    f">= 阈值 {self.correlation_threshold}"
                )
                return False
            logger.info(
                f"[DescriptorSelector] 验证通过：最大相关 {max_corr:.4f} "
                f"< 阈值 {self.correlation_threshold}"
            )
            return True
        except Exception as e:
            logger.error(f"相关性验证出错: {e}")
            return False


# ────────────────────────────────────────────────────────────────
# 兼容别名
# ────────────────────────────────────────────────────────────────

# 保留旧名作别名，方便迁移
MolecularDescriptorsCalculator = MordredFeaturizer
