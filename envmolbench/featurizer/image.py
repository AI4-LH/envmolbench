"""
分子图片特征化器。

提取自 cnn_fine.ipynb 中的 smiles_to_image() 函数，
并封装为标准 BaseFeaturizer 接口。

用途：将 SMILES 转换为分子结构图片，
供 CNNTrainer（ResNet18 等视觉模型）使用。
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .base import BaseFeaturizer

logger = logging.getLogger(__name__)


def smiles_to_mol_image(
    smiles: str,
    img_size: int = 224,
    kekulize: bool = True,
):
    """
    将单个 SMILES 字符串转换为 PIL 格式的分子结构图片（RGB，白色背景）。

    原始函数名 smiles_to_image 改为 smiles_to_mol_image，
    避免与其他库中的同名函数冲突。

    Args:
        smiles: SMILES 字符串。
        img_size: 图片边长（像素），默认 224（符合 ImageNet 预训练标准）。
        kekulize: 是否将芳香键写为 Kekulé 形式，默认 True。

    Returns:
        PIL.Image 对象（RGB）或 None（解析失败时）。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from PIL import Image

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if kekulize:
            try:
                Chem.Kekulize(mol)
            except Exception:
                pass  # Kekulize 失败时仍继续，不影响绘图

        img = Draw.MolToImage(mol, size=(img_size, img_size), fitImage=True)

        # 将 RGBA 转换为 RGB（白色背景），避免 CNN 输入出错
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except Exception as e:
        logger.debug(f"SMILES '{smiles}' 转换图片失败: {e}")
        return None


class MolImageFeaturizer(BaseFeaturizer):
    """
    分子图片特征化器：SMILES → PIL Image → 保存到临时目录。

    返回的 numpy 数组包含每个 SMILES 对应的图片文件路径（字符串）。
    CNNTrainer 读取这些路径来加载图片进行训练。

    Args:
        img_size: 图片边长，默认 224。
        save_dir: 图片保存目录；为 None 时使用系统临时目录。
    """

    def __init__(self, img_size: int = 224, save_dir: Optional[Union[str, Path]] = None):
        self.img_size = img_size
        self.save_dir = Path(save_dir) if save_dir else None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def transform(self, smiles: List[str]) -> np.ndarray:
        """
        将 SMILES 转换为分子图片，保存后返回路径数组。

        Returns:
            形状为 (n_samples,) 的字符串数组，每项为图片路径；
            转换失败的位置为空字符串 ""。
        """
        # 确定保存目录
        if self.save_dir is None:
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(prefix="envmolbench_imgs_")
            save_path = Path(self._temp_dir.name)
        else:
            save_path = self.save_dir
            save_path.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, smi in enumerate(smiles):
            img = smiles_to_mol_image(smi, img_size=self.img_size)
            if img is None:
                paths.append("")
                continue
            img_path = save_path / f"mol_{i}.png"
            img.save(img_path)
            paths.append(str(img_path))

        failed = paths.count("")
        if failed > 0:
            logger.warning(f"共 {failed}/{len(smiles)} 个 SMILES 转换图片失败。")

        return np.array(paths)

    def cleanup(self) -> None:
        """清理临时目录（如使用了临时目录）。"""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self):
        self.cleanup()

    @property
    def name(self) -> str:
        return f"mol_image_{self.img_size}"
