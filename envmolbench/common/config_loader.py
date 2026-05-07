"""
YAML 配置加载与合并模块。

加载逻辑：先读取 base.yaml 作为默认值，
再用模型专属 yaml（如 chemberta.yaml）覆盖对应字段，
最后允许通过字典参数进一步覆盖，实现三层配置合并。
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# config/ 目录默认位于本包的上层（envmolbench/config/）
_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典：override 中的值递归覆盖 base 中的同名键。
    非字典类型直接覆盖。
    """
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(
    model_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    加载配置并按三层优先级合并：
        base.yaml < {model_name}.yaml < extra（运行时覆盖）

    Args:
        model_name: 模型名称（对应 config/{model_name}.yaml）；
                    为 None 时只加载 base.yaml。
        extra: 运行时额外覆盖的字典（如 CLI 传入的参数）。
        config_dir: 配置文件目录；为 None 时使用包内默认目录。

    Returns:
        合并后的配置字典。
    """
    cfg_dir = Path(config_dir) if config_dir else _CONFIG_DIR

    # 第1层：base.yaml
    base_path = cfg_dir / "base.yaml"
    config: Dict[str, Any] = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"未找到 base.yaml（路径: {base_path}），使用空配置。")

    # 第2层：模型专属 yaml
    if model_name:
        model_path = cfg_dir / f"{model_name}.yaml"
        if model_path.exists():
            with open(model_path, "r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
            config = _deep_merge(config, model_cfg)
        else:
            logger.debug(f"未找到模型配置文件 {model_path}，跳过。")

    # 第3层：运行时覆盖
    if extra:
        config = _deep_merge(config, extra)

    return config
