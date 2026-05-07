"""
CNN 图片模型封装（ResNet18 等）。

提取自 cnn_fine.ipynb 的核心逻辑，
重构为继承 BaseModel 的 CNNTrainer 类。

原 FineFlow 类重命名为 CNNTrainer，
原 _fine_cnn() 方法重命名为 _fit_single()。

依赖：fastai, torch, rdkit, pillow
"""
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseModel
from ..featurizer.image import MolImageFeaturizer, smiles_to_mol_image

logger = logging.getLogger(__name__)

# 支持的 CNN 架构名称 → fastai/torchvision 模型
_CNN_ARCHITECTURES = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
}


class CNNTrainer(BaseModel):
    """
    基于分子结构图片的 CNN 训练器（原 FineFlow 重命名）。

    训练流程：
      1. SMILES → 分子图片（224×224，RDKit 绘制）
      2. fastai DataBlock 加载图片数据
      3. 使用 1cycle 策略微调预训练 CNN（ImageNet 权重）
      4. 测试时通过 TTA（测试时数据增强）生成预测

    Args:
        task_type: 'regression' 或 'classification'。
        arch: CNN 架构名称，默认 'resnet18'。
        img_size: 图片边长，默认 224。
        batch_size: 批大小，默认 64。
        max_epochs: 最大训练轮数，默认 3000（配合早停）。
        patience: 早停耐心值，默认 100。
        tta_n: 测试时增强次数，默认 5。
        config: 从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        task_type: str = "regression",
        arch: str = "resnet18",
        img_size: int = 224,
        batch_size: int = 64,
        max_epochs: int = 3000,
        patience: int = 100,
        tta_n: int = 5,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.arch = arch.lower()
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tta_n = tta_n

        self._learn = None  # fastai Learner 实例
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def _get_fastai_arch(self):
        """获取 fastai 对应的模型架构函数。"""
        try:
            import torchvision.models as tvm
            arch_fn = getattr(tvm, self.arch, None)
            if arch_fn is None:
                raise ValueError(f"不支持的 CNN 架构: '{self.arch}'，可用: {list(_CNN_ARCHITECTURES.keys())}")
            return arch_fn
        except ImportError:
            raise ImportError("CNNTrainer 需要安装 torchvision。")

    def _prepare_image_df(self, smiles: List[str], labels: Optional[np.ndarray], split_flag: int) -> pd.DataFrame:
        """
        将 SMILES 转换为图片并生成 DataFrame（包含图片路径和标签）。

        split_flag: 0 = 训练集（is_valid=False），1 = 验证/测试集（is_valid=True）。
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="envmolbench_cnn_")

        records = []
        for i, smi in enumerate(smiles):
            img = smiles_to_mol_image(smi, img_size=self.img_size)
            if img is None:
                continue
            img_path = Path(self._temp_dir.name) / f"mol_{split_flag}_{i}.png"
            img.save(img_path)
            record = {"image_path": str(img_path), "is_valid": bool(split_flag)}
            if labels is not None:
                record["label"] = float(labels[i])
            records.append(record)

        return pd.DataFrame(records)

    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "CNNTrainer":
        try:
            from fastai.vision.all import (
                DataBlock, ImageBlock, RegressionBlock, CategoryBlock,
                ColReader, ColSplitter, DataLoaders,
                aug_transforms, Normalize, imagenet_stats,
                vision_learner, SaveModelCallback, EarlyStoppingCallback,
            )
            import torch
        except ImportError as e:
            raise ImportError(f"CNNTrainer 需要安装 fastai: {e}")

        logger.info(
            f"训练 CNNTrainer（{self.task_type}）：arch={self.arch}，"
            f"样本={len(train_smiles)}"
        )

        # 生成图片 DataFrame
        train_df = self._prepare_image_df(train_smiles, train_labels, split_flag=0)
        if val_smiles:
            val_df = self._prepare_image_df(val_smiles, val_labels, split_flag=1)
            df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            df = train_df

        if df.empty:
            raise ValueError("无有效分子可转换为图片，CNNTrainer 无法训练。")

        # 数据增强
        base_tfms = aug_transforms(do_flip=False, max_rotate=360, pad_mode="border")

        # 构建 DataBlock
        if self.task_type == "regression":
            block_y = RegressionBlock()
        else:
            block_y = CategoryBlock()

        dblock = DataBlock(
            blocks=(ImageBlock, block_y),
            get_x=ColReader("image_path"),
            get_y=ColReader("label"),
            splitter=ColSplitter("is_valid"),
            batch_tfms=[*base_tfms, Normalize.from_stats(*imagenet_stats)],
        )
        dls = dblock.dataloaders(df, bs=self.batch_size)

        # 构建 Learner（fastai）
        arch_fn = self._get_fastai_arch()
        self._learn = vision_learner(dls, arch_fn, metrics=[])

        # 1-cycle 训练策略（自动 lr_find）
        try:
            suggested_lr = self._learn.lr_find(suggest_funcs=["valley"]).valley
        except Exception:
            suggested_lr = 1e-3

        callbacks = [
            SaveModelCallback(monitor="valid_loss", fname="best_cnn"),
            EarlyStoppingCallback(monitor="valid_loss", patience=self.patience),
        ]

        self._fit_single(suggested_lr, callbacks)
        self._is_fitted = True
        logger.info("CNNTrainer 训练完成。")
        return self

    def _fit_single(self, lr: float, callbacks: list) -> None:
        """
        执行单次微调（原 _fine_cnn() 重命名）。
        先解冻最后几层，再全量微调。
        """
        self._learn.fine_tune(
            self.max_epochs,
            base_lr=lr,
            cbs=callbacks,
        )

    def predict(self, smiles: List[str]) -> np.ndarray:
        self._check_fitted()
        try:
            from fastai.vision.all import DataBlock, ImageBlock, RegressionBlock, CategoryBlock, ColReader, GrandparentSplitter, DataLoaders, Normalize, imagenet_stats
        except ImportError as e:
            raise ImportError(f"预测需要安装 fastai: {e}")

        # 生成测试图片
        test_df = self._prepare_image_df(smiles, None, split_flag=1)
        if test_df.empty:
            return np.full(len(smiles), np.nan, dtype=np.float32)

        test_dl = self._learn.dls.test_dl(test_df["image_path"].tolist())

        # TTA 预测（测试时数据增强）
        preds, _ = self._learn.tta(dl=test_dl, n=self.tta_n)
        preds = preds.numpy().flatten().astype(np.float32)

        # 将结果映射回原始 SMILES 顺序（部分 SMILES 可能无效）
        result = np.full(len(smiles), np.nan, dtype=np.float32)
        valid_count = min(len(preds), len(smiles))
        result[:valid_count] = preds[:valid_count]
        return result

    def save(self, path: Union[str, Path]) -> None:
        if self._learn is None:
            raise RuntimeError("模型尚未训练，无法保存。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._learn.export(path)
        logger.info(f"CNN 模型已保存到 {path}")

    def load(self, path: Union[str, Path]) -> "CNNTrainer":
        try:
            from fastai.vision.all import load_learner
        except ImportError:
            raise ImportError("加载 CNN 模型需要安装 fastai。")
        self._learn = load_learner(path)
        self._is_fitted = True
        return self

    def __del__(self):
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass

    @property
    def name(self) -> str:
        return f"cnn_{self.arch}"
