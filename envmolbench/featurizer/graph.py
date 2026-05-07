"""
图特征化器（用于 GNN 模型）。

迁移自 gnn_suite/data.py，将 SMILES 转换为 torch_geometric.data.Data 对象列表，
而非返回 numpy 矩阵（GNN 需要图结构）。

BaseFeaturizer 的 transform() 返回 numpy 数组，
GraphFeaturizer 重载为返回 Data 对象列表，因此使用 transform_to_graphs() 作为主方法。
"""
import logging
import collections
from typing import List, Optional, Tuple, Union

import numpy as np

from .base import BaseFeaturizer

logger = logging.getLogger(__name__)


def get_atom_tokens(smiles: str) -> List[str]:
    """从 SMILES 中提取原子符号列表（加氢后）。"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        mol = Chem.AddHs(mol)
        return [atom.GetSymbol() for atom in mol.GetAtoms()]
    except Exception:
        return []


class AtomVocabulary:
    """
    为 SMILES 中的原子符号构建词汇表（索引映射）。

    迁移自 gnn_suite/data.py 的 SmilesVocabulary，重命名更准确。
    """

    def __init__(
        self,
        token_lists: List[List[str]],
        min_freq: int = 0,
        reserved: List[str] = None,
    ):
        if reserved is None:
            reserved = ["<unk>"]

        all_tokens = [t for lst in token_lists for t in lst]
        counter = collections.Counter(all_tokens)
        self.idx_to_token = (
            list(sorted({t for t, f in counter.items() if f >= min_freq})) + reserved
        )
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(t) for t in tokens]
        return self.token_to_idx.get(tokens, self.unk_idx)

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx["<unk>"]


# 保留旧名作别名
SmilesVocabulary = AtomVocabulary


class GraphFeaturizer(BaseFeaturizer):
    """
    将 SMILES 转换为 GNN 所需的分子图（torch_geometric.data.Data）。

    依赖：`pip install torch-geometric`

    必须先调用 fit() 以构建词汇表，然后才能调用 transform_to_graphs()。
    """

    def __init__(self, use_edge_attr: bool = True):
        self.use_edge_attr = use_edge_attr
        self._vocab: Optional[AtomVocabulary] = None

    def fit(self, smiles: List[str], labels=None) -> "GraphFeaturizer":
        """从训练集 SMILES 构建原子词汇表。"""
        token_lists = [get_atom_tokens(smi) for smi in smiles]
        self._vocab = AtomVocabulary(token_lists)
        logger.info(f"GraphFeaturizer 词汇表构建完成，共 {len(self._vocab)} 种原子。")
        return self

    def transform(self, smiles: List[str]) -> np.ndarray:
        """
        返回图索引数组（每个元素为在 transform_to_graphs 结果中的位置）。

        注意：GNN 实际使用 transform_to_graphs()，此方法仅为满足 BaseFeaturizer 接口。
        """
        raise NotImplementedError(
            "GraphFeaturizer 请使用 transform_to_graphs(smiles, labels) 方法，"
            "而非 transform()。"
        )

    def transform_to_graphs(self, smiles: List[str], labels: List[float]):
        """
        将 SMILES 列表转换为 torch_geometric.data.Data 对象列表。

        Args:
            smiles: SMILES 字符串列表。
            labels: 对应的标签列表（float）。

        Returns:
            Data 对象列表，无法解析的 SMILES 对应位置为 None。
        """
        if self._vocab is None:
            raise RuntimeError("请先调用 fit() 以构建词汇表。")

        try:
            import torch
            from rdkit import Chem
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError(f"GraphFeaturizer 需要 torch 和 torch-geometric: {e}")

        symbol_list = self._vocab.idx_to_token

        def _one_hot_unk(x, allowable):
            if x not in allowable:
                x = allowable[-1]
            return [1 if s == x else 0 for s in allowable]

        def _atom_features(atom) -> "torch.Tensor":
            from rdkit.Chem import rdchem
            hybridization_types = [
                rdchem.HybridizationType.SP,
                rdchem.HybridizationType.SP2,
                rdchem.HybridizationType.SP3,
                rdchem.HybridizationType.SP3D,
                rdchem.HybridizationType.SP3D2,
                "UNSPECIFIED",
            ]
            feats = (
                _one_hot_unk(atom.GetSymbol(), symbol_list)
                + _one_hot_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
                + _one_hot_unk(atom.GetHybridization(), hybridization_types)
                + [1 if atom.GetIsAromatic() else 0]
                + _one_hot_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
                + [1 if atom.IsInRing() else 0]
            )
            return torch.tensor(feats, dtype=torch.float)

        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
            Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
            Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
            Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
        }

        graphs = []
        for smi, label in zip(smiles, labels):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(f"无法解析 SMILES '{smi}'，跳过。")
                graphs.append(None)
                continue

            x = torch.stack([_atom_features(atom) for atom in mol.GetAtoms()])
            edge_indices, edge_attrs = [], []

            for bond in mol.GetBonds():
                s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bfeats = bond_type_map.get(bond.GetBondType(), [0, 0, 0, 0])
                edge_indices.extend([[s, e], [e, s]])
                edge_attrs.extend([bfeats, bfeats])

            edge_index = (
                torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                if edge_indices
                else torch.empty((2, 0), dtype=torch.long)
            )
            edge_attr = (
                torch.tensor(edge_attrs, dtype=torch.float)
                if edge_attrs
                else torch.empty((0, 4), dtype=torch.float)
            )
            y = torch.tensor([label], dtype=torch.float)

            if self.use_edge_attr:
                graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            else:
                graphs.append(Data(x=x, edge_index=edge_index, y=y))

        return graphs

    def get_node_feature_dim(self) -> int:
        """返回节点特征维度（由词汇表大小和固定特征数决定）。"""
        if self._vocab is None:
            raise RuntimeError("请先调用 fit()。")
        # 词汇表大小 + degree(6) + charge(2) + hybridization(6) + aromatic(1) + numHs(5) + inRing(1)
        return len(self._vocab) + 6 + 2 + 6 + 1 + 5 + 1

    @property
    def name(self) -> str:
        return "graph"


def create_gnn_dataloaders(graph_list, split_indices, batch_size: int = 32, num_workers: int = 0):
    """
    根据索引划分创建 GNN DataLoader 三元组。

    Args:
        graph_list: Data 对象列表（含 None 时会被过滤）。
        split_indices: (train_idx, val_idx, test_idx) 元组。
        batch_size: 批大小，默认 32。
        num_workers: DataLoader 工作进程数，默认 0。

    Returns:
        (train_loader, val_loader, test_loader) 元组，test_loader 可为 None。
    """
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        raise ImportError("create_gnn_dataloaders 需要安装 torch-geometric。")

    train_idx, val_idx, test_idx = split_indices

    def _filter(indices):
        return [graph_list[i] for i in indices if graph_list[i] is not None]

    train_loader = DataLoader(_filter(train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(_filter(val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = (
        DataLoader(_filter(test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if test_idx
        else None
    )
    return train_loader, val_loader, test_loader
