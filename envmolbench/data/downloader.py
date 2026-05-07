"""
数据集下载模块。

从项目官网下载内置数据集到本地。

用法示例::

    import envmolbench as eb

    # 下载单个数据集
    eb.download_dataset("hlm")                    # 保存到默认目录 ~/.envmolbench/datasets/
    eb.download_dataset("hlm", save_dir="/data")  # 指定目录

    # 下载全部 45 个数据集
    eb.download_all_datasets()
    eb.download_all_datasets(save_dir="/data/envmolbench")
"""
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from .registry import DATASET_REGISTRY

_BASE_URL = "https://www.ai4env.world/envmolbench/datasets/"
_DEFAULT_DIR = Path.home() / ".envmolbench" / "datasets"


def download_dataset(
    name: str,
    save_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    从官网下载单个内置数据集到本地。

    Args:
        name:      数据集名称（与 load_dataset() 中的 name 一致）。
        save_dir:  保存目录；None 时使用 ~/.envmolbench/datasets/。
        overwrite: True 时强制重新下载，即使文件已存在。

    Returns:
        下载后 CSV 文件的 Path 对象。
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集 '{name}'。可用名称: {sorted(DATASET_REGISTRY.keys())}"
        )

    filename = DATASET_REGISTRY[name]
    dest_dir = Path(save_dir) if save_dir else _DEFAULT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists() and not overwrite:
        print(f"[envmolbench] 文件已存在，跳过下载: {dest_path}")
        return dest_path

    _download_file(_BASE_URL + filename, dest_path, desc=filename)
    return dest_path


def download_all_datasets(
    save_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    从官网逐个下载全部 45 个内置数据集。

    Args:
        save_dir:  保存目录；None 时使用 ~/.envmolbench/datasets/。
        overwrite: True 时强制重新下载已存在的文件。

    Returns:
        数据集保存目录的 Path 对象。
    """
    dest_dir = Path(save_dir) if save_dir else _DEFAULT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = len(DATASET_REGISTRY)
    print(f"[envmolbench] 开始下载 {total} 个数据集到: {dest_dir}")
    failed = []
    for name, filename in DATASET_REGISTRY.items():
        dest_path = dest_dir / filename
        if dest_path.exists() and not overwrite:
            continue
        try:
            _download_file(_BASE_URL + filename, dest_path, desc=f"  {name}")
        except Exception as exc:
            print(f"  [警告] {name} 下载失败: {exc}")
            failed.append(name)

    if failed:
        print(f"[envmolbench] 以下数据集下载失败: {failed}")
    else:
        print(f"[envmolbench] 全部数据集下载完成: {dest_dir}")

    return dest_dir


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """带进度条的流式文件下载。"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc, leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
