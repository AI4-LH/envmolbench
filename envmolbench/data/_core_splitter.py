"""
核心数据划分类（私有模块）。

完整迁移自 cpu_ml_gnn/data_splitter.py。
外部请通过 envmolbench.data.splitter.split_data() 调用，
不要直接导入此模块。
"""
import itertools
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# 默认参数（原 cpu_ml_gnn/config.py 中的常量）
_DEFAULT_VALID_SIZE = 0.1
_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_RANDOM_SEED = 42


class DataSplitter:
    """
    封装多种分子数据集划分策略的类。

    支持的策略：
      - scaffold：基于 Murcko 骨架
      - random：随机划分（支持分层）
      - time：基于时间戳的顺序划分
      - butina：基于 Butina 化学聚类
      - maxmin：基于 MaxMin 多样性选择

    每种策略均支持单次划分（train/val/test）和交叉验证（CV）模式，
    并通过 pickle 缓存加速重复调用。
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        smiles_col: str = "smiles",
        label_col: str = "label",
        time_col: Optional[str] = None,
        cache_path: Optional[Union[str, Path]] = None,
    ):
        self.dataset = dataset
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.time_col = time_col
        self.cache_path = Path(cache_path) if cache_path else None

        self.indices = np.arange(len(self.dataset))
        self.labels = self.dataset[label_col].values

        if smiles_col not in self.dataset.columns:
            logger.warning(f"'{smiles_col}' 列不存在，基于结构的划分方法将不可用。")
            self.smiles = None
        else:
            self.smiles = self.dataset[smiles_col].tolist()

        if time_col and time_col not in self.dataset.columns:
            raise ValueError(f"指定的时间列 '{time_col}' 在数据集中不存在。")

        self._scaffold_sets: Optional[List[List[int]]] = None
        self._morgan_fps_cache: Dict[tuple, tuple] = {}
        self._butina_cluster_sets_cache: Dict[tuple, List[List[int]]] = {}
        self._cache_data: dict = self._load_cache()

    # ── 缓存管理 ─────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"从 {self.cache_path} 加载划分缓存。")
                return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
        return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._cache_data, f)
            except Exception as e:
                logger.error(f"保存缓存失败: {e}")

    # ── Scaffold 划分 ────────────────────────────────────────────────────────

    @staticmethod
    def _generate_murcko_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
        """为单个 SMILES 生成 Murcko 骨架字符串。"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
            return scaffold if scaffold else ""
        except Exception as e:
            logger.error(f"为 '{smiles}' 生成骨架时出错: {e}")
            return None

    def _get_scaffold_sets(self) -> List[List[int]]:
        if self._scaffold_sets is not None:
            return self._scaffold_sets
        if self.smiles is None:
            raise ValueError("SMILES 列未提供，无法生成骨架。")

        scaffolds: Dict[str, List[int]] = {}
        failed = 0
        for idx, smi in enumerate(self.smiles):
            scaffold = self._generate_murcko_scaffold(smi)
            if scaffold is not None:
                scaffolds.setdefault(scaffold, []).append(idx)
            else:
                failed += 1

        if failed:
            logger.warning(f"{failed} 个分子未能生成骨架。")

        scaffolds = {k: sorted(v) for k, v in scaffolds.items()}
        self._scaffold_sets = [
            sset
            for _, sset in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]
        logger.info(f"生成 {len(self._scaffold_sets)} 个骨架组。")
        return self._scaffold_sets

    def get_scaffold_train_val_test_split(
        self,
        valid_size: float = _DEFAULT_VALID_SIZE,
        test_size: float = _DEFAULT_TEST_SIZE,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Scaffold 单次划分（train/val/test）。"""
        key = f"scaffold_single_{valid_size}_{test_size}"
        if key in self._cache_data:
            return self._cache_data[key]

        scaffold_sets = self._get_scaffold_sets()
        n = len(self.dataset)
        train_cutoff = int((1 - valid_size - test_size) * n)
        valid_cutoff = int((1 - test_size) * n)

        train_inds, valid_inds, test_inds = [], [], []
        for sset in scaffold_sets:
            if len(train_inds) + len(sset) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(sset) > valid_cutoff:
                    test_inds += sset
                else:
                    valid_inds += sset
            else:
                train_inds += sset

        split = (sorted(train_inds), sorted(valid_inds), sorted(test_inds))
        self._cache_data[key] = split
        self._save_cache()
        return split

    def get_scaffold_cv_splits(
        self,
        n_splits: int = 5,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """Scaffold 交叉验证划分。"""
        key = f"scaffold_cv_{n_splits}_{test_size}_{seed}"
        if key in self._cache_data:
            return self._cache_data[key]

        scaffold_sets = self._get_scaffold_sets()
        random.seed(seed)
        shuffled = random.sample(scaffold_sets, len(scaffold_sets))

        test_cutoff = int(test_size * len(self.dataset))
        test_inds: List[int] = []
        train_val_sets: List[List[int]] = []
        cur_size = 0
        for sset in shuffled:
            if cur_size + len(sset) <= test_cutoff:
                test_inds.extend(sset)
                cur_size += len(sset)
            else:
                train_val_sets.append(sset)

        folds: List[List[int]] = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        for sset in train_val_sets:
            min_idx = min(range(n_splits), key=lambda i: fold_sizes[i])
            folds[min_idx].extend(sset)
            fold_sizes[min_idx] += len(sset)

        splits = []
        for i in range(n_splits):
            train = list(itertools.chain.from_iterable(folds[j] for j in range(n_splits) if j != i))
            splits.append((sorted(train), sorted(folds[i]), sorted(test_inds)))

        self._cache_data[key] = splits
        self._save_cache()
        return splits

    # ── Random 划分 ──────────────────────────────────────────────────────────

    def get_random_train_val_test_split(
        self,
        valid_size: float = _DEFAULT_VALID_SIZE,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Random 单次划分（支持分层）。"""
        key = f"random_single_{valid_size}_{test_size}_{seed}"
        if key in self._cache_data:
            return self._cache_data[key]

        is_cls = len(np.unique(self.labels)) < 10
        stratify = self.labels if is_cls else None

        train_val, test = train_test_split(
            self.indices, test_size=test_size, random_state=seed, stratify=stratify
        )
        relative_val = valid_size / (1.0 - test_size)
        stratify2 = self.labels[train_val] if is_cls else None
        train, val = train_test_split(
            train_val, test_size=relative_val, random_state=seed, stratify=stratify2
        )

        split = (sorted(train.tolist()), sorted(val.tolist()), sorted(test.tolist()))
        self._cache_data[key] = split
        self._save_cache()
        return split

    def get_random_cv_splits(
        self,
        n_splits: int = 5,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """Random 交叉验证划分。"""
        key = f"random_cv_{n_splits}_{test_size}_{seed}"
        if key in self._cache_data:
            return self._cache_data[key]

        is_cls = len(np.unique(self.labels)) < 10
        stratify = self.labels if is_cls else None

        train_val, test = train_test_split(
            self.indices, test_size=test_size, random_state=seed, stratify=stratify
        )
        y_tv = self.labels[train_val]
        kf = StratifiedKFold(n_splits, shuffle=True, random_state=seed) if is_cls else KFold(n_splits, shuffle=True, random_state=seed)

        splits = []
        for tr_rel, val_rel in kf.split(train_val, y_tv):
            splits.append((
                sorted(train_val[tr_rel].tolist()),
                sorted(train_val[val_rel].tolist()),
                sorted(test.tolist()),
            ))

        self._cache_data[key] = splits
        self._save_cache()
        return splits

    # ── Time 划分 ────────────────────────────────────────────────────────────

    def get_time_train_val_test_split(
        self,
        valid_size: float = _DEFAULT_VALID_SIZE,
        test_size: float = _DEFAULT_TEST_SIZE,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Time 单次划分。"""
        if not self.time_col:
            raise ValueError("Time 划分需要指定 time_col。")
        key = f"time_single_{valid_size}_{test_size}"
        if key in self._cache_data:
            return self._cache_data[key]

        sorted_idx = self.dataset.sort_values(by=self.time_col).index.to_numpy()
        n = len(sorted_idx)
        test_cut = int(n * (1 - test_size))
        val_cut = int(n * (1 - test_size - valid_size))

        split = (
            sorted(sorted_idx[:val_cut].tolist()),
            sorted(sorted_idx[val_cut:test_cut].tolist()),
            sorted(sorted_idx[test_cut:].tolist()),
        )
        self._cache_data[key] = split
        self._save_cache()
        return split

    def get_time_cv_splits(
        self,
        n_splits: int = 5,
        test_size: float = _DEFAULT_TEST_SIZE,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """Time 交叉验证划分。"""
        if not self.time_col:
            raise ValueError("Time 划分需要指定 time_col。")
        key = f"time_cv_{n_splits}_{test_size}"
        if key in self._cache_data:
            return self._cache_data[key]

        sorted_idx = self.dataset.sort_values(by=self.time_col).index.to_numpy()
        n = len(sorted_idx)
        test_cut = int(n * (1 - test_size))
        test_inds = sorted_idx[test_cut:]
        tv_inds = sorted_idx[:test_cut]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for tr_rel, val_rel in tscv.split(tv_inds):
            splits.append((
                sorted(tv_inds[tr_rel].tolist()),
                sorted(tv_inds[val_rel].tolist()),
                sorted(test_inds.tolist()),
            ))

        self._cache_data[key] = splits
        self._save_cache()
        return splits

    # ── Butina 划分 ──────────────────────────────────────────────────────────

    def _compute_morgan_fps(
        self, radius: int = 2, n_bits: int = 2048
    ) -> Tuple[list, List[int]]:
        """计算并缓存所有分子的 Morgan 指纹。"""
        if self.smiles is None:
            raise ValueError("SMILES 列未提供。")
        cache_key = (radius, n_bits)
        if cache_key in self._morgan_fps_cache:
            return self._morgan_fps_cache[cache_key]

        fps, valid_idx = [], []
        failed = 0
        for idx, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed += 1
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
            valid_idx.append(idx)

        if failed:
            logger.warning(f"计算 Morgan 指纹时 {failed} 个分子解析失败。")
        self._morgan_fps_cache[cache_key] = (fps, valid_idx)
        return fps, valid_idx

    def _get_butina_cluster_sets(
        self, threshold: float = 0.4, radius: int = 2, n_bits: int = 2048
    ) -> Tuple[List[List[int]], List[int]]:
        cache_key = (threshold, radius, n_bits)
        if cache_key in self._butina_cluster_sets_cache:
            fps, valid_idx = self._compute_morgan_fps(radius, n_bits)
            return self._butina_cluster_sets_cache[cache_key], valid_idx

        fps, valid_idx = self._compute_morgan_fps(radius, n_bits)
        n = len(fps)
        dist_matrix = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dist_matrix.extend(1.0 - s for s in sims)

        raw = Butina.ClusterData(dist_matrix, n, 1.0 - threshold, isDistData=True, reordering=False)
        cluster_sets = [
            sorted(valid_idx[li] for li in cluster) for cluster in raw
        ]
        cluster_sets.sort(key=lambda cs: (len(cs), cs[0]), reverse=True)
        self._butina_cluster_sets_cache[cache_key] = cluster_sets
        return cluster_sets, valid_idx

    def get_butina_train_val_test_split(
        self,
        valid_size: float = _DEFAULT_VALID_SIZE,
        test_size: float = _DEFAULT_TEST_SIZE,
        tanimoto_threshold: float = 0.4,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Butina 聚类单次划分。"""
        key = f"butina_single_{valid_size}_{test_size}_{tanimoto_threshold}_{morgan_radius}_{morgan_n_bits}"
        if key in self._cache_data:
            return self._cache_data[key]

        clusters, _ = self._get_butina_cluster_sets(tanimoto_threshold, morgan_radius, morgan_n_bits)
        n = len(self.dataset)
        train_cut = int((1 - valid_size - test_size) * n)
        valid_cut = int((1 - test_size) * n)

        train_inds, valid_inds, test_inds = [], [], []
        for cs in clusters:
            if len(train_inds) + len(cs) > train_cut:
                if len(train_inds) + len(valid_inds) + len(cs) > valid_cut:
                    test_inds += cs
                else:
                    valid_inds += cs
            else:
                train_inds += cs

        split = (sorted(train_inds), sorted(valid_inds), sorted(test_inds))
        self._cache_data[key] = split
        self._save_cache()
        return split

    def get_butina_cv_splits(
        self,
        n_splits: int = 5,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
        tanimoto_threshold: float = 0.4,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """Butina 聚类交叉验证划分。"""
        key = f"butina_cv_{n_splits}_{test_size}_{seed}_{tanimoto_threshold}_{morgan_radius}_{morgan_n_bits}"
        if key in self._cache_data:
            return self._cache_data[key]

        clusters, _ = self._get_butina_cluster_sets(tanimoto_threshold, morgan_radius, morgan_n_bits)
        random.seed(seed)
        shuffled = random.sample(clusters, len(clusters))

        test_cut = int(test_size * len(self.dataset))
        test_inds: List[int] = []
        tv_sets: List[List[int]] = []
        cur = 0
        for cs in shuffled:
            if cur + len(cs) <= test_cut:
                test_inds.extend(cs)
                cur += len(cs)
            else:
                tv_sets.append(cs)

        folds: List[List[int]] = [[] for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        for cs in tv_sets:
            mi = min(range(n_splits), key=lambda i: fold_sizes[i])
            folds[mi].extend(cs)
            fold_sizes[mi] += len(cs)

        splits = []
        for i in range(n_splits):
            train = list(itertools.chain.from_iterable(folds[j] for j in range(n_splits) if j != i))
            splits.append((sorted(train), sorted(folds[i]), sorted(test_inds)))

        self._cache_data[key] = splits
        self._save_cache()
        return splits

    # ── MaxMin 划分 ──────────────────────────────────────────────────────────

    def get_maxmin_train_val_test_split(
        self,
        valid_size: float = _DEFAULT_VALID_SIZE,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> Tuple[List[int], List[int], List[int]]:
        """MaxMin 多样性选择单次划分。"""
        key = f"maxmin_single_{valid_size}_{test_size}_{seed}_{morgan_radius}_{morgan_n_bits}"
        if key in self._cache_data:
            return self._cache_data[key]

        fps, valid_idx = self._compute_morgan_fps(morgan_radius, morgan_n_bits)
        n = len(fps)
        n_test = max(1, int(test_size * len(self.dataset)))
        n_val = max(1, int(valid_size * len(self.dataset)))

        if n_test + n_val >= n:
            logger.warning("数据集过小，MaxMin 划分降级为顺序分配。")
            split = (
                sorted(valid_idx[i] for i in range(n_test + n_val, n)),
                sorted(valid_idx[i] for i in range(n_test, n_test + n_val)),
                sorted(valid_idx[i] for i in range(n_test)),
            )
            self._cache_data[key] = split
            self._save_cache()
            return split

        picker = rdSimDivPickers.MaxMinPicker()

        def _dist(i, j):
            return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

        test_local = list(picker.LazyPick(_dist, n, n_test, firstPicks=[], seed=seed))
        test_set = set(test_local)

        remaining_local = [i for i in range(n) if i not in test_set]
        rem_fps = [fps[i] for i in remaining_local]
        n_rem = len(rem_fps)

        def _dist_rem(i, j, _fps=rem_fps):
            return 1.0 - DataStructs.TanimotoSimilarity(_fps[i], _fps[j])

        val_in_rem = list(picker.LazyPick(_dist_rem, n_rem, n_val, firstPicks=[], seed=seed))
        val_set = set(val_in_rem)

        split = (
            sorted(valid_idx[remaining_local[i]] for i in range(n_rem) if i not in val_set),
            sorted(valid_idx[remaining_local[i]] for i in val_in_rem),
            sorted(valid_idx[i] for i in test_local),
        )
        self._cache_data[key] = split
        self._save_cache()
        return split

    def get_maxmin_cv_splits(
        self,
        n_splits: int = 5,
        test_size: float = _DEFAULT_TEST_SIZE,
        seed: int = _DEFAULT_RANDOM_SEED,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048,
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """MaxMin 多样性选择交叉验证划分。"""
        key = f"maxmin_cv_{n_splits}_{test_size}_{seed}_{morgan_radius}_{morgan_n_bits}"
        if key in self._cache_data:
            return self._cache_data[key]

        fps, valid_idx = self._compute_morgan_fps(morgan_radius, morgan_n_bits)
        n = len(fps)
        n_test = max(1, int(test_size * len(self.dataset)))

        picker = rdSimDivPickers.MaxMinPicker()

        def _dist(i, j):
            return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

        test_local = list(picker.LazyPick(_dist, n, n_test, firstPicks=[], seed=seed))
        test_set_local = set(test_local)
        test_orig = sorted(valid_idx[i] for i in test_local)

        pool_local = [i for i in range(n) if i not in test_set_local]
        pool_fps = [fps[i] for i in pool_local]
        n_pool = len(pool_fps)
        n_val_per_fold = n_pool // n_splits

        splits = []
        for fold_idx in range(n_splits):
            if n_val_per_fold > 0:
                _pfps = pool_fps

                def _dist_pool(i, j, _fps=_pfps):
                    return 1.0 - DataStructs.TanimotoSimilarity(_fps[i], _fps[j])

                val_in_pool = list(picker.LazyPick(
                    _dist_pool, n_pool, n_val_per_fold, firstPicks=[], seed=seed + fold_idx
                ))
                val_set = set(val_in_pool)
            else:
                val_in_pool = []
                val_set = set()

            train_in_pool = [i for i in range(n_pool) if i not in val_set]
            splits.append((
                sorted(valid_idx[pool_local[i]] for i in train_in_pool),
                sorted(valid_idx[pool_local[i]] for i in val_in_pool),
                test_orig,
            ))

        self._cache_data[key] = splits
        self._save_cache()
        return splits
