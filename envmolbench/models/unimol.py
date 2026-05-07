"""
Uni-Mol 模型封装。

提取自 train_unimol.ipynb 的核心训练逻辑，
重构为继承 BaseModel 的 UnimolTrainer 类。

依赖：unimol_tools
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseModel

logger = logging.getLogger(__name__)

# 动态训练参数（根据数据集规模）
_DYNAMIC_PARAMS = [
    {"min": 0,      "max": 1_000,  "epochs": 50, "batch_size": 16, "lr": 1e-4},
    {"min": 1_000,  "max": 10_000, "epochs": 40, "batch_size": 32, "lr": 5e-5},
    {"min": 10_000, "max": float("inf"), "epochs": 30, "batch_size": 64, "lr": 1e-5},
]


def _select_dynamic_params(n: int, config: Dict) -> Dict:
    for p in _DYNAMIC_PARAMS:
        if p["min"] <= n < p["max"]:
            return {
                "epochs": config.get("epochs", p["epochs"]),
                "batch_size": config.get("batch_size", p["batch_size"]),
                "lr": config.get("lr", p["lr"]),
            }
    return {"epochs": 30, "batch_size": 64, "lr": 1e-5}


class UnimolTrainer(BaseModel):
    """
    Uni-Mol 3D 构象模型训练器。

    Uni-Mol 使用 3D 分子构象（通过 RDKit 从 SMILES 生成），
    对空间结构更敏感，通常在含 3D 信息的任务上表现优异。

    训练策略：
      1. 将 train/val 数据写入临时 CSV 文件
      2. 尝试将 Scaffold val 集注入 MolTrain（需 unimol_tools 源码修改支持）
      3. 若 unimol_tools 不支持 valid_data 参数，自动回退到内部随机划分

    Args:
        task_type: 'regression' 或 'classification'。
        freeze_layers: 冻结的底部层数，默认 6（迁移学习）。
        conformer_num: 生成的 3D 构象数量，默认 1（平衡速度与精度）。
        config: 从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        task_type: str = "regression",
        freeze_layers: int = 6,
        conformer_num: int = 1,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.freeze_layers = freeze_layers
        self.conformer_num = conformer_num

        self._clf = None
        self._temp_files: List[str] = []

    def _write_temp_csv(self, smiles: List[str], labels: Optional[np.ndarray], suffix: str) -> str:
        """将 SMILES 和标签写入临时 CSV，返回文件路径。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{suffix}.csv", delete=False, encoding="utf-8"
        ) as f:
            path = f.name

        df = pd.DataFrame({"smiles": smiles})
        if labels is not None:
            df["target"] = labels
        df.to_csv(path, index=False)
        self._temp_files.append(path)
        return path

    def _cleanup_temps(self) -> None:
        """清理所有临时 CSV 文件。"""
        for fp in self._temp_files:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
        self._temp_files.clear()

    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "UnimolTrainer":
        try:
            from unimol_tools import MolTrain
        except ImportError:
            raise ImportError(
                "UnimolTrainer 需要安装 unimol_tools：pip install unimol_tools"
            )

        n = len(train_smiles)
        params = _select_dynamic_params(n, self.config)
        logger.info(
            f"训练 Uni-Mol（{self.task_type}）：样本={n}，"
            f"epochs={params['epochs']}，batch={params['batch_size']}"
        )

        # 写入临时数据文件
        train_path = self._write_temp_csv(train_smiles, train_labels, "train")
        val_path = self._write_temp_csv(val_smiles, val_labels, "val") if val_smiles else None

        # 构建 MolTrain 参数
        mol_train_kwargs = {
            "task": self.task_type,
            "data_type": "molecule",
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "learning_rate": params["lr"],
            "conformer_num": self.conformer_num,
            "num_workers": self.config.get("num_workers", 4),
            "use_amp": self.config.get("use_amp", True),
            "freeze_layers": self.freeze_layers,
            "patience": self.config.get("patience", 5),
        }

        self._clf = MolTrain(**mol_train_kwargs)

        # 尝试使用外部 Scaffold 验证集
        # 注意：标准 unimol_tools 不支持 valid_data 参数；
        # 若源码已修改以支持外部验证集，此处将生效；否则自动回退。
        if val_path:
            try:
                self._clf.fit(train_path, valid_data=val_path)
                logger.info("成功使用外部 Scaffold 验证集监控训练。")
            except TypeError:
                logger.warning(
                    "当前 unimol_tools 版本不支持 valid_data 参数，"
                    "回退到内部随机划分模式。如需使用外部验证集，"
                    "请参考文档修改 unimol_tools 源码。"
                )
                # 合并 train+val 后交由 unimol 内部划分
                combined_smiles = train_smiles + (val_smiles or [])
                combined_labels = np.concatenate([train_labels, val_labels if val_labels is not None else []])
                combined_path = self._write_temp_csv(combined_smiles, combined_labels, "combined")
                self._clf.fit(combined_path)
        else:
            self._clf.fit(train_path)

        self._is_fitted = True
        self._cleanup_temps()
        logger.info("Uni-Mol 训练完成。")
        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        self._check_fitted()
        try:
            from unimol_tools import MolPredict
        except ImportError:
            raise ImportError("预测需要安装 unimol_tools。")

        test_path = self._write_temp_csv(smiles, None, "test")
        try:
            clf_pred = MolPredict(load_model=self._clf.save_path)
            raw = clf_pred.predict(data_path=test_path)

            # 统一处理输出格式（dict 或 array）
            if isinstance(raw, dict):
                preds = raw.get("target", list(raw.values())[0])
            else:
                preds = raw

            result = np.array(preds, dtype=np.float32).flatten()
        finally:
            self._cleanup_temps()

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Uni-Mol 由 unimol_tools 自动管理模型保存，此方法记录保存路径。"""
        if self._clf is not None and hasattr(self._clf, "save_path"):
            logger.info(f"Uni-Mol 模型已由 unimol_tools 保存到: {self._clf.save_path}")
        else:
            logger.warning("Uni-Mol 模型尚未保存或路径不可用。")

    def load(self, path: Union[str, Path]) -> "UnimolTrainer":
        """从已保存的 Uni-Mol 模型路径加载预测器。"""
        try:
            from unimol_tools import MolPredict
        except ImportError:
            raise ImportError("加载 Uni-Mol 模型需要安装 unimol_tools。")
        self._load_path = str(path)
        self._is_fitted = True
        return self

    def __del__(self):
        self._cleanup_temps()

    @property
    def name(self) -> str:
        return "unimol"
