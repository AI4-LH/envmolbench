"""
envmolbench —— 分子属性预测基准测试框架

提供统一的数据加载、特征化、建模和评估接口，
支持 47 个环境分子数据集与 10+ 种模型（传统 ML、GNN、Transformer、CNN 等）。

快速开始::

    # 方式1：分步调用（灵活组合）
    from envmolbench.data import load_dataset, split_data
    from envmolbench.models import get_model
    from envmolbench.common.metrics import calc_metrics

    smiles, labels, task = load_dataset("esol")
    splits = split_data(smiles, labels, method="scaffold")
    model = get_model("chemprop", task_type=task)
    model.fit(splits.train_smiles, splits.train_labels,
              splits.val_smiles,   splits.val_labels)
    preds = model.predict(splits.test_smiles)
    print(calc_metrics(splits.test_labels, preds, task))

    # 方式2：一键训练
    from envmolbench import quick_train
    results = quick_train(model="chemprop", dataset="esol", split="scaffold")

    # 方式3：查看可用选项
    from envmolbench import list_datasets, list_models, list_featurizers
    print(list_datasets())
    print(list_models())
    print(list_featurizers())
"""

__version__ = "0.1.2"
__author__ = "EnvMolBench Team"

# ── 便捷导入：将最常用的公共 API 提升到顶层 ─────────────────────────

from .data import load_dataset, split_data, list_datasets, download_dataset, download_all_datasets
from .models import get_model, list_models
from .featurizer import get_featurizer, list_featurizers
from .common.metrics import calc_metrics

# ── Roughness analysis ────────────────────────────────────────────────────────
from .roughness import roughness, compute_roughness, compute_nndr, compute_sali

# ── Conformal prediction / uncertainty calibration ────────────────────────────
from .conformal import (
    conformal_prediction,
    compute_ece,
    conformal_regression,
    conformal_classification,
    compute_ece_regression,
    compute_ece_classification,
)


# ── quick_train：一键训练快捷入口 ────────────────────────────────────

from typing import Optional


def quick_train(
    model: str,
    dataset: str,
    split: str = "scaffold",
    task: Optional[str] = None,
    datasets_dir: Optional[str] = None,
    result_csv: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """
    一键训练：加载数据集、划分、训练、评估，返回指标字典。

    Args:
        model:           模型名称（如 'chemprop', 'rf', 'gnn'）。
        dataset:         数据集名称（内置47个）或 CSV 文件路径。
        split:           数据划分方法（'scaffold'/'random'/'time'/'butina'/'maxmin'）。
                         默认 'scaffold'。
        task:            任务类型（'regression'/'classification'）；
                         None 时自动检测。
        datasets_dir:    数据集目录；None 时使用默认路径。
        result_csv:      结果保存路径；None 时不保存文件。
        config_overrides:额外配置参数字典，会覆盖 YAML 配置。

    Returns:
        包含 train/val/test 各指标的字典。

    Examples::

        results = quick_train("chemprop", "esol")
        results = quick_train("rf", "my_data.csv", split="random", task="regression")
    """
    from .pipeline.runner import PipelineRunner
    from .common.config_loader import load_config

    config = load_config(model, extra=config_overrides)
    runner = PipelineRunner(config=config, result_csv=result_csv)
    return runner.run(
        model_name=model,
        dataset_name=dataset,
        split_method=split,
        task_type=task,
        datasets_dir=datasets_dir,
    )


# ── 公共 API 列表 ─────────────────────────────────────────────────────

__all__ = [
    # 顶层快捷函数
    "quick_train",
    # 数据层
    "load_dataset",
    "split_data",
    "list_datasets",
    "download_dataset",
    "download_all_datasets",
    # 模型层
    "get_model",
    "list_models",
    # 特征化层
    "get_featurizer",
    "list_featurizers",
    # 评估
    "calc_metrics",
    # 粗糙度分析
    "roughness",
    "compute_roughness",
    "compute_nndr",
    "compute_sali",
    # 不确定性校准
    "conformal_prediction",
    "compute_ece",
    "conformal_regression",
    "conformal_classification",
    "compute_ece_regression",
    "compute_ece_classification",
    # 版本
    "__version__",
]
