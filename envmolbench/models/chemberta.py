"""
ChemBERTa 模型封装。

提取自 chemberta.ipynb 的核心训练逻辑，
重构为继承 BaseModel 的 ChembertaTrainer 类。

依赖：transformers, torch
预训练模型：
  - 分类：DeepChem/ChemBERTa-77M-MLM
  - 回归：DeepChem/ChemBERTa-77M-MTR
"""
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)

# 预训练模型路径（HuggingFace 模型 ID）
_PRETRAINED = {
    "classification": "DeepChem/ChemBERTa-77M-MLM",
    "regression": "DeepChem/ChemBERTa-77M-MTR",
}

# 动态训练参数：根据数据集规模自动选择（<1K / 1K-10K / >10K）
_DYNAMIC_PARAMS = [
    {"min": 0,    "max": 1_000,  "batch_size": 16, "epochs": 40, "lr": 1e-5},
    {"min": 1_000, "max": 10_000, "batch_size": 32, "epochs": 30, "lr": 2e-5},
    {"min": 10_000, "max": float("inf"), "batch_size": 64, "epochs": 15, "lr": 3e-5},
]


def _select_dynamic_params(n_samples: int, config: Dict) -> Dict:
    """根据样本量自动选择训练超参数，可被 config 覆盖。"""
    for params in _DYNAMIC_PARAMS:
        if params["min"] <= n_samples < params["max"]:
            return {
                "batch_size": config.get("batch_size", params["batch_size"]),
                "epochs": config.get("epochs", params["epochs"]),
                "lr": config.get("lr", params["lr"]),
            }
    return {"batch_size": 64, "epochs": 15, "lr": 3e-5}


class ChembertaTrainer(BaseModel):
    """
    ChemBERTa 微调训练器。

    训练流程：
      1. 自动检测 SMILES 长度分布，选择 max_len（64/128/256/512）
      2. 回归任务对标签做 StandardScaler 归一化
      3. 使用 AdamW + 线性预热学习率调度
      4. 早停（patience=5，监控验证集损失）
      5. 推理时反归一化输出

    Args:
        task_type: 'regression' 或 'classification'。
        pretrained_model: 预训练模型 ID 或本地路径；None 时使用默认。
        linear_probe: 是否冻结编码器，只训练分类头（线性探针模式）。
        config: 从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        task_type: str = "regression",
        pretrained_model: Optional[str] = None,
        linear_probe: bool = False,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.pretrained_model = pretrained_model or _PRETRAINED[task_type]
        self.linear_probe = linear_probe

        self._model = None
        self._tokenizer = None
        self._scaler = None  # 回归任务的标签标准化器（保存以供推理用）
        self._device = None
        self._max_len: int = 128

    def _get_device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _detect_max_len(self, smiles: List[str]) -> int:
        """基于第 95 百分位 SMILES 长度自动选择 max_len。"""
        lengths = [len(s) for s in smiles]
        p95 = int(np.percentile(lengths, 95))
        for thresh in [64, 128, 256]:
            if p95 <= thresh:
                return thresh
        return 512

    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "ChembertaTrainer":
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import LinearLR
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(f"ChembertaTrainer 需要安装 transformers 和 torch: {e}")

        n = len(train_smiles)
        params = _select_dynamic_params(n, self.config)
        self._device = self._get_device()
        self._max_len = self._detect_max_len(train_smiles)

        logger.info(
            f"训练 ChemBERTa（{self.task_type}）：样本={n}，max_len={self._max_len}，"
            f"batch={params['batch_size']}，epochs={params['epochs']}，lr={params['lr']}"
        )

        # 标签归一化（仅回归任务）
        y_train = train_labels.astype(np.float32)
        y_val = val_labels.astype(np.float32) if val_labels is not None else None

        if self.task_type == "regression":
            self._scaler = StandardScaler()
            y_train = self._scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            if y_val is not None:
                y_val = self._scaler.transform(y_val.reshape(-1, 1)).flatten()

        # 加载 tokenizer 和模型
        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        num_labels = 1  # 二分类和回归均使用单输出
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ).to(self._device)

        # 线性探针模式：冻结编码器
        if self.linear_probe:
            for param in self._model.base_model.parameters():
                param.requires_grad = False
            logger.info("线性探针模式：编码器已冻结，只训练分类头。")

        # 构建 Dataset
        class _MolDataset(Dataset):
            def __init__(self, smi_list, lbl_array, tokenizer, max_len):
                self.smi = smi_list
                self.lbl = lbl_array
                self.tok = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.smi)

            def __getitem__(self, idx):
                enc = self.tok(
                    self.smi[idx],
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(self.lbl[idx], dtype=torch.float32),
                }

        train_ds = _MolDataset(train_smiles, y_train, self._tokenizer, self._max_len)
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader = None
        if val_smiles and y_val is not None:
            val_ds = _MolDataset(val_smiles, y_val, self._tokenizer, self._max_len)
            val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

        # 优化器与学习率调度（线性预热 2 轮）
        optimizer = AdamW(
            self._model.parameters(),
            lr=params["lr"],
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        warmup_epochs = 2
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

        loss_fn = (
            nn.BCEWithLogitsLoss()
            if self.task_type == "classification"
            else nn.MSELoss()
        )

        patience = self.config.get("patience", 5)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(params["epochs"]):
            # 训练
            self._model.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["label"].to(self._device)
                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            if epoch < warmup_epochs:
                scheduler.step()

            # 验证与早停
            if val_loader:
                val_loss = self._eval_loss(val_loader, loss_fn)
                logger.debug(f"Epoch {epoch + 1}/{params['epochs']} | 验证损失: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    import copy
                    best_state = copy.deepcopy(self._model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"早停于第 {epoch + 1} 轮。")
                        break

        # 恢复最佳模型
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("ChemBERTa 训练完成。")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return self

    def _eval_loss(self, loader, loss_fn) -> float:
        """计算验证集平均损失。"""
        import torch
        self._model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["label"].to(self._device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                total_loss += loss_fn(logits, labels).item() * len(labels)
                n += len(labels)
        return total_loss / n if n > 0 else float("inf")

    def predict(self, smiles: List[str]) -> np.ndarray:
        self._check_fitted()
        import torch
        from torch.utils.data import Dataset, DataLoader

        class _PredDataset(Dataset):
            def __init__(self, smi_list, tokenizer, max_len):
                self.smi = smi_list
                self.tok = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.smi)

            def __getitem__(self, idx):
                enc = self.tok(
                    self.smi[idx],
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                }

        ds = _PredDataset(smiles, self._tokenizer, self._max_len)
        loader = DataLoader(ds, batch_size=64)

        self._model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                if self.task_type == "classification":
                    preds = torch.sigmoid(logits).cpu().numpy()
                else:
                    preds = logits.cpu().numpy()
                all_preds.append(preds)

        result = np.concatenate(all_preds).flatten().astype(np.float32)

        # 回归：反归一化
        if self.task_type == "regression" and self._scaler is not None:
            result = self._scaler.inverse_transform(result.reshape(-1, 1)).flatten().astype(np.float32)

        return result

    def save(self, path: Union[str, Path]) -> None:
        import torch, pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"ChemBERTa 模型已保存到 {path}")

    def load(self, path: Union[str, Path]) -> "ChembertaTrainer":
        import torch, pickle
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = Path(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._device = self._get_device()
        self._model.to(self._device)
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
        self._is_fitted = True
        return self

    @property
    def name(self) -> str:
        return "chemberta"
