"""
通用工具函数模块。

迁移自 cpu_ml_gnn/utils.py，并补充中文注释，
移除已迁移到其他模块的日志配置函数（见 common/logger.py）。
"""
import csv
import logging
import time
import os
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from filelock import FileLock
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# ─── 子进程辅助（用于超时控制）───────────────────────────────────────────────

def _subprocess_wrapper(queue: Queue, target_func: Callable, args: tuple, kwargs: dict) -> None:
    """
    顶层函数（可被 pickle 序列化），在子进程中执行 target_func，
    并将结果或异常放入 queue。
    """
    try:
        result = target_func(*args, **kwargs)
        queue.put(result)
    except Exception as exc:
        queue.put(exc)


def run_with_timeout(
    func: Callable,
    args: tuple,
    kwargs: dict,
    timeout_seconds: Optional[int],
) -> Any:
    """
    在带有超时保护的子进程中运行函数。

    Args:
        func: 要执行的目标函数。
        args: 函数位置参数（元组）。
        kwargs: 函数关键字参数（字典）。
        timeout_seconds: 超时秒数；为 None 时直接在当前进程运行。

    Returns:
        函数返回值。

    Raises:
        TimeoutError: 超过 timeout_seconds 仍未完成。
        Exception: 子进程中抛出的任何异常。
    """
    # 无超时需求时省去子进程开销
    if timeout_seconds is None:
        return func(*args, **kwargs)

    queue: Queue = Queue()
    proc = Process(target=_subprocess_wrapper, args=(queue, func, args, kwargs))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        # 获取函数真实名称（兼容 functools.partial）
        real_func = getattr(func, "func", func)
        func_name = getattr(real_func, "__name__", "unknown_function")
        logger.warning(f"函数 '{func_name}' 执行超时 (>{timeout_seconds}s)，正在终止进程。")
        proc.terminate()
        proc.join()
        raise TimeoutError(f"函数 '{func_name}' 执行超时")

    result = queue.get()
    if isinstance(result, Exception):
        logger.error(f"子进程中发生异常: {result}")
        raise result

    return result


# ─── 路径管理 ──────────────────────────────────────────────────────────────

def build_output_paths(
    results_root: Union[str, Path],
    data_name: str,
    split_type: str,
    cv: int,
    iteration: int = 0,
) -> Dict[str, Path]:
    """
    生成标准化的输出文件路径字典。

    Args:
        results_root: 根结果目录（如 'results/'）。
        data_name: 数据集名称（不含扩展名）。
        split_type: 数据划分类型（如 'scaffold'）。
        cv: 交叉验证折数。
        iteration: 实验迭代编号，默认 0。

    Returns:
        包含各类输出路径的字典。
    """
    root = Path(results_root)
    subdir = root / data_name
    subdir.mkdir(parents=True, exist_ok=True)

    return {
        "results_dir": subdir,
        "baseline_csv": subdir / f"baseline_metrics_{iteration}.csv",
        "hyperopt_iter_csv": subdir / f"hyperopt_iterations_{iteration}.csv",
        "best_params_csv": subdir / f"best_hyperparameters_{iteration}.csv",
        "split_cache_pkl": subdir / f"split_indices_{split_type}_{cv}_{data_name}.pkl",
    }


# ─── 超参数处理 ────────────────────────────────────────────────────────────

def clean_hyperparameters(
    raw_params: Dict,
    model_name: str,
    feature_name: str,
    integer_params: List[str],
    models_needing_scaler: List[str],
    features_needing_scaler: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    清理和准备从超参数搜索获取的参数字典。

    处理内容：
    1. 提取 scaler 类型（仅对需要缩放的模型+特征组合）
    2. 解析 LogisticRegression 的结构化 solver 参数
    3. 将整数类型参数强制转换为 int
    4. 清理 SVR/SVC 的 kernel 不兼容参数

    Returns:
        (清理后的参数字典, scaler 类型字符串或 None)
    """
    params = raw_params.copy()
    scaler_type: Optional[str] = None

    # 提取或删除 scaler 参数
    if model_name in models_needing_scaler and feature_name in features_needing_scaler:
        scaler_type = params.pop("scaler", None)
    else:
        params.pop("scaler", None)

    # 解析 LogisticRegression 的结构化 solver 参数
    if model_name == "LogisticRegression" and "solver_config" in params:
        solver_cfg = params.pop("solver_config")
        params["solver"] = solver_cfg["solver"]
        params.update(solver_cfg.get("penalty_config", {}))

    # 整数参数转换
    for param_name in integer_params:
        if param_name in params and params[param_name] is not None:
            try:
                params[param_name] = int(float(params[param_name]))
            except (ValueError, TypeError):
                logger.warning(f"无法将参数 '{param_name}' 转换为整数，已移除。")
                del params[param_name]

    # 清理 SVR/SVC 的 kernel 不兼容参数
    if model_name in ("SVR", "SVC"):
        kernel = params.get("kernel", "rbf")
        if kernel != "poly":
            params.pop("degree", None)
        if kernel not in ("rbf", "poly", "sigmoid"):
            params.pop("gamma", None)
        if kernel not in ("poly", "sigmoid"):
            params.pop("coef0", None)

    return params, scaler_type


# ─── CSV 写入 ──────────────────────────────────────────────────────────────

def append_to_csv(
    filepath: Union[str, Path],
    data_dict: Dict,
    header: List[str],
) -> None:
    """
    安全地追加一行数据到 CSV 文件（带文件锁，防止并发写冲突）。

    Args:
        filepath: 目标 CSV 文件路径。
        data_dict: 要写入的数据字典，键对应 header。
        header: CSV 的列名列表。
    """
    filepath = Path(filepath)
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")

    try:
        with FileLock(str(lock_path), timeout=10):
            file_exists = filepath.exists() and filepath.stat().st_size > 0
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_dict)
    except Exception as e:
        logger.error(f"写入 CSV 文件 {filepath} 失败: {e}", exc_info=True)


# ─── 数据集工具 ────────────────────────────────────────────────────────────

def get_datasets_sorted_by_size(
    folder: Union[str, Path],
    smiles_col: str = "smiles",
    ascending: bool = True,
) -> Dict[str, int]:
    """
    扫描目录下所有 CSV 文件，按行数排序并返回名称→行数字典。

    Args:
        folder: CSV 文件所在目录。
        smiles_col: 用于读取行数的列名。
        ascending: True 为升序（小数据集优先）。

    Returns:
        {数据集名称: 行数} 的有序字典。
    """
    name_to_rows: Dict[str, int] = {}
    for csv_path in Path(folder).glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, usecols=[smiles_col])
            name_to_rows[csv_path.stem] = len(df)
        except Exception:
            continue

    return dict(
        sorted(name_to_rows.items(), key=lambda item: item[1], reverse=not ascending)
    )


def calc_zero_sparsity(X: np.ndarray) -> float:
    """计算特征矩阵的零值稀疏度（零值比例）。"""
    density = np.count_nonzero(X) / X.size
    return 1.0 - density


# ─── 特征缩放 ──────────────────────────────────────────────────────────────

_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
}


def apply_feature_scaling(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    scaler_type: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    """
    在 CPU 上对特征矩阵进行标准化/归一化。

    缩放器仅在训练集上 fit，然后 transform 验证集和测试集，
    防止数据泄露。

    Args:
        X_train: 训练特征矩阵。
        X_val: 验证特征矩阵（可为 None）。
        X_test: 测试特征矩阵（可为 None）。
        scaler_type: 缩放类型（'standard' / 'minmax' / 'maxabs' / 'robust'）；
                     None 或 'passthrough' 时跳过缩放。

    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)
        其中 fitted_scaler 可用于保存以便推理时复用。
    """
    if scaler_type is None or scaler_type.lower() in ("passthrough", "none"):
        return X_train, X_val, X_test, None

    scaler_class = _SCALER_MAP.get(scaler_type.lower())
    if scaler_class is None:
        logger.warning(f"未知的缩放器类型 '{scaler_type}'，跳过缩放。")
        return X_train, X_val, X_test, None

    logger.info(f"应用特征缩放器: {scaler_type}")
    scaler = scaler_class()

    try:
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val) if X_val is not None else None
        X_test_s = scaler.transform(X_test) if X_test is not None else None
    except Exception as e:
        logger.error(f"应用缩放器 '{scaler_type}' 时出错: {e}，返回原始数据。", exc_info=True)
        return X_train, X_val, X_test, None

    return X_train_s, X_val_s, X_test_s, scaler
