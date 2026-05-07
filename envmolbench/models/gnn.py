"""
GNN 模型封装（GCN / GNN）。

合并自 gnn_suite/models.py（模型架构）+ gnn_suite/training.py（训练逻辑），
重构为统一的 GNNModel 类，继承 BaseModel。

依赖：torch, torch-geometric
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseModel
from ..featurizer.graph import GraphFeaturizer, create_gnn_dataloaders

logger = logging.getLogger(__name__)


# ─── 模型架构定义 ─────────────────────────────────────────────────────────────

def _build_gnn_architecture(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
    edge_feature_dim: int,
):
    """
    工厂函数：根据 model_type 构建 GCN 或 GNN（支持边特征）。
    迁移自 gnn_suite/models.py 的 GCN 和 GNN 类。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.nn import Sequential as Seq, Linear, ReLU, Dropout
        from torch_geometric.nn import BatchNorm, global_add_pool, GraphConv, MessagePassing
    except ImportError as e:
        raise ImportError(f"GNN 模型需要安装 torch 和 torch-geometric: {e}")

    class _CustomConv(MessagePassing):
        """自定义 GNN 卷积层，可利用边特征。"""
        def __init__(self, in_ch, out_ch, edge_dim):
            super().__init__(aggr="add")
            self.mlp = Seq(
                Linear(in_ch + edge_dim, out_ch),
                ReLU(),
                Linear(out_ch, out_ch),
            )

        def forward(self, x, edge_index, edge_attr):
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        def message(self, x_j, edge_attr):
            return self.mlp(torch.cat([x_j, edge_attr], dim=1))

    class _GNN(torch.nn.Module):
        """支持边特征的多层 GNN，逐层递减隐藏维度。"""
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropout_rate = dropout

            dims = [input_dim]
            cur = hidden_dim
            for _ in range(num_layers):
                dims.append(max(cur, 16))
                cur //= 2

            for i in range(num_layers):
                self.convs.append(_CustomConv(dims[i], dims[i + 1], edge_feature_dim))
                self.batch_norms.append(BatchNorm(dims[i + 1]))

            last = dims[-1]
            self.pool = global_add_pool
            self.mlp = Seq(
                Linear(last, max(last // 2, 16)),
                ReLU(),
                Dropout(p=dropout),
                Linear(max(last // 2, 16), output_dim),
            )

        def forward(self, x, edge_index, batch, edge_attr=None):
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index, torch.zeros(edge_index.size(1), edge_feature_dim, device=x.device))
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            return self.mlp(self.pool(x, batch))

    class _GCN(torch.nn.Module):
        """不使用边特征的多层图卷积网络（GraphConv）。"""
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.dropout_rate = dropout

            dims = [input_dim]
            cur = hidden_dim
            for _ in range(num_layers):
                dims.append(max(cur, 16))
                cur //= 2

            for i in range(num_layers):
                self.convs.append(GraphConv(dims[i], dims[i + 1]))
                self.batch_norms.append(BatchNorm(dims[i + 1]))

            last = dims[-1]
            self.pool = global_add_pool
            self.mlp = Seq(
                Linear(last, max(last // 2, 16)),
                ReLU(),
                Dropout(p=dropout),
                Linear(max(last // 2, 16), output_dim),
            )

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.batch_norms):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            return self.mlp(self.pool(x, batch))

    if model_type.lower() == "gcn":
        return _GCN()
    else:
        return _GNN()


# ─── 早停辅助类 ───────────────────────────────────────────────────────────────

class _EarlyStopping:
    """训练早停，迁移自 gnn_suite/training.py。"""

    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, value: float, model) -> None:
        is_better = (
            self.best_value is None
            or (self.mode == "min" and value < self.best_value)
            or (self.mode == "max" and value > self.best_value)
        )
        if is_better:
            self.best_value = value
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ─── GNNModel 封装 ────────────────────────────────────────────────────────────

class GNNModel(BaseModel):
    """
    GNN/GCN 模型封装类。

    完整流程：
      1. fit() → GraphFeaturizer.fit(train_smiles) 构建词汇表
      2. 转换 SMILES → 图列表 (Data 对象)
      3. 按 split_indices 创建 DataLoader
      4. 训练循环（含早停）

    Args:
        model_type: 'gnn'（带边特征）或 'gcn'（不带边特征），默认 'gnn'。
        task_type: 'regression' 或 'classification'。
        hidden_dim: GNN 隐藏层维度，默认 256。
        num_layers: GNN 层数，默认 3。
        dropout: Dropout 率，默认 0.1。
        num_epochs: 最大训练轮数，默认 100。
        patience: 早停耐心值，默认 10。
        lr: 学习率，默认 1e-3。
        batch_size: 批大小，默认 32。
        config: 从 YAML 加载的配置字典。
    """

    def __init__(
        self,
        model_type: str = "gnn",
        task_type: str = "regression",
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_epochs: int = 100,
        patience: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        config: Optional[Dict] = None,
    ):
        super().__init__(task_type, config)
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size

        self._featurizer = GraphFeaturizer(use_edge_attr=(model_type.lower() == "gnn"))
        self._model = None
        self._device = None

    def _get_device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train_smiles: List[str],
        train_labels: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> "GNNModel":
        try:
            import torch
            import torch.nn as nn
            from sklearn.metrics import root_mean_squared_error, log_loss
        except ImportError as e:
            raise ImportError(f"GNNModel 需要安装 torch 和 scikit-learn: {e}")

        self._device = self._get_device()
        logger.info(f"训练 {self.model_type.upper()}（{self.task_type}），设备: {self._device}")

        # 构建词汇表并生成图
        all_smiles = train_smiles + (val_smiles or [])
        all_labels = list(train_labels) + list(val_labels if val_labels is not None else [])
        self._featurizer.fit(all_smiles)

        train_graphs = self._featurizer.transform_to_graphs(train_smiles, train_labels.tolist())
        val_graphs = (
            self._featurizer.transform_to_graphs(val_smiles, val_labels.tolist())
            if val_smiles
            else []
        )

        # 过滤无效图
        train_graphs = [g for g in train_graphs if g is not None]
        val_graphs = [g for g in val_graphs if g is not None]

        if not train_graphs:
            raise ValueError("训练集中无有效分子，无法训练 GNN。")

        # 创建 DataLoader
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.batch_size) if val_graphs else None

        # 构建模型
        node_dim = self._featurizer.get_node_feature_dim()
        edge_dim = 4  # bond type one-hot（SINGLE/DOUBLE/TRIPLE/AROMATIC）
        self._model = _build_gnn_architecture(
            model_type=self.model_type,
            input_dim=node_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout,
            edge_feature_dim=edge_dim,
        ).to(self._device)

        # 初始化权重
        for m in self._model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss() if self.task_type == "classification" else nn.MSELoss()
        early_stopping = _EarlyStopping(patience=self.patience, mode="min")

        # 训练循环
        for epoch in range(self.num_epochs):
            self._model.train()
            for data in train_loader:
                data = data.to(self._device)
                optimizer.zero_grad()
                has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None
                out = (
                    self._model(data.x, data.edge_index, data.batch, data.edge_attr)
                    if has_edge_attr
                    else self._model(data.x, data.edge_index, data.batch)
                )
                loss = loss_fn(out, data.y.view_as(out))
                loss.backward()
                optimizer.step()

            # 验证与早停
            if val_loader:
                y_pred, y_true = self._predict_loader(val_loader)
                val_metric = (
                    log_loss(y_true, y_pred)
                    if self.task_type == "classification"
                    else root_mean_squared_error(y_true, y_pred)
                )
                early_stopping(val_metric, self._model)
                if early_stopping.early_stop:
                    logger.info(f"早停于第 {epoch + 1} 轮，最佳验证指标: {early_stopping.best_value:.4f}")
                    break

        # 恢复最佳模型状态
        if early_stopping.best_state is not None:
            self._model.load_state_dict(early_stopping.best_state)

        self._is_fitted = True
        logger.info(f"{self.model_type.upper()} 训练完成。")
        return self

    def _predict_loader(self, loader) -> tuple:
        """对 DataLoader 中的图批次生成预测，返回 (y_pred, y_true)。"""
        import torch
        self._model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self._device)
                has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None
                out = (
                    self._model(data.x, data.edge_index, data.batch, data.edge_attr)
                    if has_edge_attr
                    else self._model(data.x, data.edge_index, data.batch)
                )
                if self.task_type == "classification":
                    preds = torch.sigmoid(out).cpu().numpy()
                else:
                    preds = out.cpu().numpy()
                all_preds.append(preds)
                all_trues.append(data.y.cpu().numpy())

        return (
            np.concatenate(all_preds).flatten(),
            np.concatenate(all_trues).flatten(),
        )

    def predict(self, smiles: List[str]) -> np.ndarray:
        self._check_fitted()
        # 用占位标签（0）构建图，不影响节点/边特征
        graphs = self._featurizer.transform_to_graphs(smiles, [0.0] * len(smiles))
        valid = [(i, g) for i, g in enumerate(graphs) if g is not None]

        result = np.full(len(smiles), np.nan, dtype=np.float32)
        if not valid:
            return result

        from torch_geometric.loader import DataLoader
        valid_indices, valid_graphs = zip(*valid)
        loader = DataLoader(list(valid_graphs), batch_size=self.batch_size)
        preds, _ = self._predict_loader(loader)
        for idx, pred in zip(valid_indices, preds):
            result[idx] = pred

        return result

    def save(self, path: Union[str, Path]) -> None:
        import torch, pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict() if self._model else None,
            "featurizer": self._featurizer,
            "config": {
                "model_type": self.model_type,
                "task_type": self.task_type,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }, path)
        logger.info(f"GNN 模型已保存到 {path}")

    def load(self, path: Union[str, Path]) -> "GNNModel":
        import torch
        checkpoint = torch.load(path, weights_only=False)
        self._featurizer = checkpoint["featurizer"]
        cfg = checkpoint["config"]
        self.model_type = cfg["model_type"]
        self.task_type = cfg["task_type"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        if checkpoint["model_state"]:
            node_dim = self._featurizer.get_node_feature_dim()
            self._model = _build_gnn_architecture(
                self.model_type, node_dim, self.hidden_dim, 1,
                self.num_layers, self.dropout, 4
            ).to(self._get_device())
            self._model.load_state_dict(checkpoint["model_state"])
        self._is_fitted = True
        return self

    @property
    def name(self) -> str:
        return self.model_type.lower()
