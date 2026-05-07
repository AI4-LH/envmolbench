"""
envmolbench.data — 数据层

暴露数据加载、划分、清洗和数据集注册表的公共 API。

用法：
    from envmolbench.data import load_dataset, split_data, list_datasets
    from envmolbench.data import clean_chemical_data, standardize_smiles_series
"""
from .loader import load_dataset
from .splitter import split_data, SplitResult, SPLIT_METHODS
from .registry import list_datasets, DATASET_REGISTRY
from .cleaner import clean_chemical_data, standardize_smiles_series, standardize_smiles
from .downloader import download_dataset, download_all_datasets

__all__ = [
    "load_dataset",
    "split_data",
    "SplitResult",
    "SPLIT_METHODS",
    "list_datasets",
    "DATASET_REGISTRY",
    # 数据清洗
    "clean_chemical_data",
    "standardize_smiles_series",
    "standardize_smiles",
    # 数据集下载
    "download_dataset",
    "download_all_datasets",
]
