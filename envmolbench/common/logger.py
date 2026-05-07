"""
统一日志配置模块。

提供 get_logger() 工厂函数，替代各处散落的 setup_logger()，
确保格式、编码和处理器配置的一致性。
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    获取或创建一个配置好的日志记录器。

    Args:
        name: 日志器名称，通常传入 __name__。
        log_file: 日志文件路径；为 None 时只输出到控制台。
        level: 日志级别字符串，如 'INFO'、'DEBUG'。

    Returns:
        已配置好的 logging.Logger 实例。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 使用命名的 logger，避免污染 root logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 若该 logger 已有 handler，直接返回，防止重复添加
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler（始终添加）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # 可选：设置上海时区（Linux/macOS 有效，Windows 忽略）
    os.environ.setdefault("TZ", "Asia/Shanghai")
    try:
        time.tzset()
    except AttributeError:
        pass

    return logger
