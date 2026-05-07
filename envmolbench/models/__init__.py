"""
envmolbench.models — 模型层

提供工厂函数 get_model() 和模型注册表。

支持的模型：
  深度学习：
    - 'chemberta' : ChemBERTa Transformer（SMILES 序列）
    - 'chemprop'  : Chemprop MPNN（分子图）
    - 'unimol'    : Uni-Mol 3D 构象模型
    - 'cnn'       : ResNet18 等 CNN（分子图片）
    - 'gnn'       : 自定义 GNN（带边特征）
    - 'gcn'       : 图卷积网络（不含边特征）

  传统机器学习：
    - 'rf' / 'random_forest' : 随机森林
    - 'xgboost'              : XGBoost
    - 'catboost'             : CatBoost
    - 'svr' / 'svc'         : 支持向量机
    - 'ridge'                : 岭回归
    - 'lasso'                : Lasso 回归
    - 'logistic_regression'  : 逻辑回归
"""
from .base import BaseModel
from .classical import ClassicalModel
from .gnn import GNNModel
from .chemberta import ChembertaTrainer
from .chemprop import ChempropTrainer
from .unimol import UnimolTrainer
from .cnn import CNNTrainer

# 注册表：模型名称 → (类, 是否为深度学习模型)
_MODEL_REGISTRY = {
    # 深度学习模型
    "chemberta":   (ChembertaTrainer, True),
    "chemprop":    (ChempropTrainer, True),
    "unimol":      (UnimolTrainer, True),
    "cnn":         (CNNTrainer, True),
    "gnn":         (GNNModel, True),
    "gcn":         (GNNModel, True),
    # 传统机器学习（通过 ClassicalModel 统一封装）
    "rf":              (ClassicalModel, False),
    "random_forest":   (ClassicalModel, False),
    "xgboost":         (ClassicalModel, False),
    "catboost":        (ClassicalModel, False),
    "svr":             (ClassicalModel, False),
    "svc":             (ClassicalModel, False),
    "ridge":           (ClassicalModel, False),
    "lasso":           (ClassicalModel, False),
    "logistic_regression": (ClassicalModel, False),
}


def list_models() -> list:
    """返回所有已注册模型的名称列表。"""
    return sorted(_MODEL_REGISTRY.keys())


def get_model(model_name: str, task_type: str = "regression", **kwargs) -> BaseModel:
    """
    工厂函数：按名称创建模型实例。

    Args:
        model_name: 模型名称（见 list_models()）。
        task_type: 'regression' 或 'classification'。
        **kwargs: 传递给模型构造函数的额外参数。

    Returns:
        对应的 BaseModel 子类实例。

    Raises:
        ValueError: model_name 不在注册表中。
    """
    entry = _MODEL_REGISTRY.get(model_name.lower())
    if entry is None:
        raise ValueError(
            f"未知模型 '{model_name}'。\n可用模型: {list_models()}"
        )

    cls, is_deep = entry

    # 传统 ML 模型需要额外传入 model_name 给 ClassicalModel
    if cls is ClassicalModel:
        return cls(model_name=model_name.lower(), task_type=task_type, **kwargs)

    # GNN/GCN 需要传入 model_type
    if cls is GNNModel:
        return cls(model_type=model_name.lower(), task_type=task_type, **kwargs)

    return cls(task_type=task_type, **kwargs)


__all__ = [
    "BaseModel",
    "ClassicalModel",
    "GNNModel",
    "ChembertaTrainer",
    "ChempropTrainer",
    "UnimolTrainer",
    "CNNTrainer",
    "get_model",
    "list_models",
]
