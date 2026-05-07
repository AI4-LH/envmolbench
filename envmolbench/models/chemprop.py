"""
Chemprop MPNN 模型封装。

提取自 train_chemprop.ipynb 的核心训练逻辑，
重构为继承 BaseModel 的 ChempropTrainer 类。

依赖：chemprop>=2.0, torch, pytorch-lightning
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

# 动态训练参数（与 ChemBERTa 一致的规模策略）
_DYNAMIC_PARAMS = [
    {"min": 0,      "max": 1_000,  "batch_size": 16, "epochs": 50, "lr": 1e-4},
    {"min": 1_000,  "max": 10_000, "batch_size": 32, "epochs": 40, "lr": 5e-4},
    {"min": 10_000, "max": float("inf"), "batch_size": 64, "epochs": 30, "lr": 1e-3},
]


def _select_dynamic_params(n: int, config: Dict) -> Dict:
    for p in _DYNAMIC_PARAMS:
        if p["min"] <= n < p["max"]:
            return {
                "batch_size": config.get("batch_size", p["batch_size"]),
                "epochs": config.get("epochs", p["epochs"]),
                "lr": config.get("lr", p["lr"]),
            }
    return {"batch_size": 64, "epochs": 30, "lr": 1e-3}


class ChempropTrainer(BaseModel):
    """
    Chemprop 2.x MPNN 训练器。

    训练流程：
      1. 构建 BondMessagePassing + MeanAggregation + 任务头
      2. 可选加载预训练权重（灵活 key 映射，兼容不同版本）
      3. PyTorch Lightning 训练（ModelCheckpoint + EarlyStopping）
      4. 回归任务对标签做 StandardScaler 归一化

    Args:
        task_type: 'regression' 或 'classification'。
        pretrained_path: 预训练权重 .pt 文件路径；None 时从零开始训练。
        hidden_dim: MPNN 隐藏维度，默认 300（预训练时通常为 2048）。
        depth: 消息传递层数，默认 3。
        dropout: Dropout 率，默认 0.1。
        config: 从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        task_type: str = "regression",
        pretrained_path: Optional[str] = None,
        hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.pretrained_path = pretrained_path
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        self._model = None
        self._scaler = None
        self._trainer = None

    def _build_model(self, n_samples: int):
        """构建 Chemprop 2.x 模型。"""
        try:
            import torch
            from chemprop import nn as cpnn
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"ChempropTrainer 需要安装 chemprop>=2.0 和 torch: {e}")

        mp = cpnn.BondMessagePassing(d_h=self.hidden_dim, depth=self.depth, dropout=self.dropout)
        agg = cpnn.MeanAggregation()

        if self.task_type == "classification":
            ffn = cpnn.BinaryClassificationFFN()
        else:
            ffn = cpnn.RegressionFFN()

        model = cpnn.MPNN(message_passing=mp, agg=agg, predictor=ffn)

        # 尝试加载预训练权重
        if self.pretrained_path:
            model = self._load_pretrained_weights(model, self.pretrained_path)

        return model

    def _load_pretrained_weights(self, model, pretrained_path: str):
        """
        灵活加载预训练权重，兼容不同版本保存的 key 前缀差异。

        注意：使用 weights_only=False 是因为旧版 chemprop 保存文件包含
        自定义类对象，无法仅加载 tensor。这是已知的 chemprop 兼容性问题，
        请确保只加载可信来源的权重文件。
        """
        try:
            import torch
            # weights_only=False：兼容 chemprop 旧版保存格式
            state_dict = torch.load(pretrained_path, weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # 清理常见的 key 前缀
            cleaned = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in ("model.", "net.", "encoder.", "mpnn."):
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                cleaned[new_k] = v

            # 提取 message_passing 子模块权重
            mp_dict = {k.replace("message_passing.", ""): v for k, v in cleaned.items() if "message_passing" in k}

            missing, unexpected = model.message_passing.load_state_dict(mp_dict, strict=False)
            if missing:
                logger.warning(f"预训练权重未覆盖的键: {missing[:5]}...")
            logger.info(f"预训练权重加载成功（来自 {pretrained_path}）")
        except Exception as e:
            logger.warning(f"加载预训练权重失败，将从零开始训练: {e}")

        return model

    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "ChempropTrainer":
        try:
            import torch
            import pytorch_lightning as pl
            from chemprop import data as cpdata
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"ChempropTrainer 需要安装 chemprop>=2.0, pytorch-lightning: {e}")

        n = len(train_smiles)
        params = _select_dynamic_params(n, self.config)
        logger.info(
            f"训练 Chemprop（{self.task_type}）：样本={n}，"
            f"batch={params['batch_size']}，epochs={params['epochs']}"
        )

        # 标签归一化（回归任务）
        y_train = train_labels.astype(np.float32)
        y_val = val_labels.astype(np.float32) if val_labels is not None else None

        if self.task_type == "regression":
            self._scaler = StandardScaler()
            y_train = self._scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            if y_val is not None:
                y_val = self._scaler.transform(y_val.reshape(-1, 1)).flatten()

        # 构建 Chemprop 数据集
        train_data = cpdata.MoleculeDataset([
            cpdata.MoleculeDatapoint.from_smi(smi, [y])
            for smi, y in zip(train_smiles, y_train)
        ])
        val_data = None
        if val_smiles and y_val is not None:
            val_data = cpdata.MoleculeDataset([
                cpdata.MoleculeDatapoint.from_smi(smi, [y])
                for smi, y in zip(val_smiles, y_val)
            ])

        train_loader = cpdata.build_dataloader(train_data, batch_size=params["batch_size"], shuffle=True)
        val_loader = cpdata.build_dataloader(val_data, batch_size=params["batch_size"]) if val_data else None

        # 构建模型
        self._model = self._build_model(n)
        torch.set_float32_matmul_precision("medium")

        callbacks = []
        try:
            from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
            callbacks.append(ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1))
            if val_loader:
                callbacks.append(EarlyStopping(monitor="val_loss", patience=self.config.get("patience", 5)))
        except Exception:
            pass

        trainer = pl.Trainer(
            max_epochs=params["epochs"],
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(self._model, train_loader, val_loader)

        self._is_fitted = True
        logger.info("Chemprop 训练完成。")
        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        self._check_fitted()
        try:
            from chemprop import data as cpdata
        except ImportError as e:
            raise ImportError(f"预测需要安装 chemprop>=2.0: {e}")

        pred_data = cpdata.MoleculeDataset([
            cpdata.MoleculeDatapoint.from_smi(smi, [0.0]) for smi in smiles
        ])
        loader = cpdata.build_dataloader(pred_data, batch_size=64, shuffle=False)

        import torch
        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                preds = self._model(batch)
                all_preds.append(preds.cpu().numpy())

        result = np.concatenate(all_preds).flatten().astype(np.float32)

        if self.task_type == "regression" and self._scaler is not None:
            result = self._scaler.inverse_transform(result.reshape(-1, 1)).flatten().astype(np.float32)

        return result

    def save(self, path: Union[str, Path]) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "scaler": self._scaler,
            "config": {
                "task_type": self.task_type,
                "hidden_dim": self.hidden_dim,
                "depth": self.depth,
                "dropout": self.dropout,
            },
        }, path)
        logger.info(f"Chemprop 模型已保存到 {path}")

    def load(self, path: Union[str, Path]) -> "ChempropTrainer":
        import torch
        checkpoint = torch.load(path, weights_only=False)
        cfg = checkpoint["config"]
        self.task_type = cfg["task_type"]
        self.hidden_dim = cfg["hidden_dim"]
        self.depth = cfg["depth"]
        self.dropout = cfg["dropout"]
        self._scaler = checkpoint["scaler"]
        self._model = self._build_model(0)
        self._model.load_state_dict(checkpoint["model_state"])
        self._is_fitted = True
        return self

    @property
    def name(self) -> str:
        return "chemprop"
