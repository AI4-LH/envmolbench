"""
Molecular landscape roughness metrics.

Provides two roughness metrics and a unified high-level interface:

  - NNDR (Nearest-Neighbor Disagreement Rate) for classification datasets,
    based on the MODI framework (Golbraikh et al.).
  - SALI (Structure-Activity Landscape Index) for regression datasets.

Quick start::

    import envmolbench as eb

    smiles, labels, task = eb.load_dataset("hepato_tox")
    r = eb.roughness((smiles, labels, task))
    # {"task_type": "classification", "fp_type": "ecfp4", "metric": "NNDR", "mean": 0.31, ...}

    # Multiple fingerprints at once
    r = eb.roughness((smiles, labels, task), fp_type=["ecfp4", "maccs", "rdkit_topo", "atompair"])
    # {"ecfp4": {...}, "maccs": {...}, ...}

    # Low-level access
    nndr_mean, nndr_std, n = eb.compute_nndr(smiles, labels, fp_type="ecfp4")
    sali_mean, sali_std,  n = eb.compute_sali(smiles, labels, fp_type="ecfp4")
"""

import numpy as np

__all__ = ["roughness", "compute_roughness", "compute_nndr", "compute_sali"]

_VALID_FP_TYPES = ("ecfp4", "maccs", "rdkit_topo", "atompair")


def _get_fingerprint(mol, fp_type, nbits=2048):
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit import Chem

    if fp_type == "ecfp4":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    elif fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "rdkit_topo":
        return Chem.RDKFingerprint(mol, fpSize=nbits)
    elif fp_type == "atompair":
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
    else:
        raise ValueError(f"Unknown fp_type '{fp_type}'. Choose from: {_VALID_FP_TYPES}")


def compute_nndr(smiles_list, labels, fp_type="ecfp4", max_samples=2000, n_repeats=10, seed=42):
    """
    Compute NNDR (Nearest-Neighbor Disagreement Rate) for classification.

    Following Golbraikh et al.'s MODI framework:
        MODI = (1/K) * sum_class (N_same / N_total)
        NNDR = 1 - MODI

    For datasets > max_samples, uses stratified random subsampling repeated
    n_repeats times to estimate mean and variance.

    Parameters
    ----------
    smiles_list : list of str
    labels      : array-like, binary labels (0/1)
    fp_type     : str, one of "ecfp4", "maccs", "rdkit_topo", "atompair"
    max_samples : int, max molecules before subsampling kicks in
    n_repeats   : int, subsampling repeats for large datasets
    seed        : int

    Returns
    -------
    nndr_mean : float or None
    nndr_std  : float or None
    n_valid   : int
    """
    from rdkit import Chem, DataStructs

    rng = np.random.RandomState(seed)

    mols_fps = []
    valid_labels = []
    for smi, lab in zip(smiles_list, labels):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol:
                fp = _get_fingerprint(mol, fp_type)
                mols_fps.append(fp)
                valid_labels.append(int(float(lab)))
        except Exception:
            continue

    n = len(mols_fps)
    if n < 10:
        return None, None, n

    classes = sorted(set(valid_labels))
    if len(classes) < 2:
        return None, None, n

    def _compute_nndr_on_subset(fps_sub, labels_sub):
        n_sub = len(fps_sub)
        if n_sub < 10:
            return 1.0

        nn_agree = np.zeros(n_sub, dtype=bool)
        for i in range(n_sub):
            sims = DataStructs.BulkTanimotoSimilarity(fps_sub[i], fps_sub)
            sims[i] = -1
            nn_idx = int(np.argmax(sims))
            nn_agree[i] = (labels_sub[i] == labels_sub[nn_idx])

        modi_parts = []
        for c in classes:
            idxs = [j for j, y in enumerate(labels_sub) if y == c]
            if idxs:
                modi_parts.append(float(nn_agree[idxs].mean()))

        if not modi_parts:
            return 1.0
        return 1.0 - float(np.mean(modi_parts))

    if n <= max_samples:
        nndr = _compute_nndr_on_subset(mols_fps, valid_labels)
        return nndr, 0.0, n

    nndr_repeats = []
    labels_array = np.array(valid_labels)
    for _ in range(n_repeats):
        idx_by_class = {c: np.where(labels_array == c)[0] for c in classes}
        sampled_idx = []
        for c in classes:
            n_c = len(idx_by_class[c])
            n_sample_c = max(1, min(int(max_samples * (n_c / n)), n_c))
            sampled_idx.extend(rng.choice(idx_by_class[c], n_sample_c, replace=False))
        rng.shuffle(sampled_idx)
        fps_sub = [mols_fps[i] for i in sampled_idx]
        labels_sub = [valid_labels[i] for i in sampled_idx]
        nndr_repeats.append(_compute_nndr_on_subset(fps_sub, labels_sub))

    return float(np.mean(nndr_repeats)), float(np.std(nndr_repeats, ddof=0)), n


def compute_sali(smiles_list, labels, fp_type="ecfp4", max_pairs=100000, n_repeats=10, seed=42):
    """
    Compute mean SALI (Structure-Activity Landscape Index) for regression datasets.

        SALI(i, j) = |y_i - y_j| / (1 - sim(i, j))

    Labels are normalized to [0, 1] before computation.
    For datasets with total pairs > max_pairs, uses random subsampling.

    Parameters
    ----------
    smiles_list : list of str
    labels      : array-like, continuous activity values
    fp_type     : str, one of "ecfp4", "maccs", "rdkit_topo", "atompair"
    max_pairs   : int, max pairs to evaluate before subsampling kicks in
    n_repeats   : int, subsampling repeats for large datasets
    seed        : int

    Returns
    -------
    sali_mean : float or None
    sali_std  : float or None
    n_valid   : int
    """
    from rdkit import Chem, DataStructs

    rng = np.random.RandomState(seed)

    mols_fps = []
    valid_labels = []
    for smi, lab in zip(smiles_list, labels):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol:
                fp = _get_fingerprint(mol, fp_type)
                mols_fps.append(fp)
                valid_labels.append(float(lab))
        except Exception:
            continue

    n = len(mols_fps)
    if n < 10:
        return None, None, n

    labels_array = np.array(valid_labels)
    lab_min, lab_max = np.min(labels_array), np.max(labels_array)
    if lab_max <= lab_min:
        return 0.0, 0.0, n
    labels_norm = (labels_array - lab_min) / (lab_max - lab_min)

    total_pairs = n * (n - 1) // 2

    def _compute_sali_on_pairs(fps_sub, labels_sub):
        n_sub = len(fps_sub)
        sali_vals = []
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                sim = DataStructs.TanimotoSimilarity(fps_sub[i], fps_sub[j])
                if sim >= 1.0 - 1e-6:
                    continue
                delta = abs(labels_sub[i] - labels_sub[j])
                sali_vals.append(delta / (1.0 - sim))
        return float(np.mean(sali_vals)) if sali_vals else 0.0

    if total_pairs <= max_pairs:
        sali = _compute_sali_on_pairs(mols_fps, labels_norm.tolist())
        return sali, 0.0, n

    sali_repeats = []
    for _ in range(n_repeats):
        sampled_idx = rng.choice(n, size=min(max_pairs // 2, n), replace=False)
        if len(sampled_idx) < 2:
            sampled_idx = rng.choice(n, size=2, replace=False)
        fps_sub = [mols_fps[i] for i in sampled_idx]
        labels_sub = labels_norm[sampled_idx].tolist()
        sali_repeats.append(_compute_sali_on_pairs(fps_sub, labels_sub))

    return float(np.mean(sali_repeats)), float(np.std(sali_repeats, ddof=0)), n


def roughness(dataset, task_type="auto", fp_type="ecfp4", max_samples=2000, n_repeats=10, seed=42):
    """
    Compute landscape roughness from a loaded dataset.

    Accepts the output of ``eb.load_dataset()`` directly, or any 2/3-tuple.

    Args:
        dataset:     A 3-tuple ``(smiles, labels, task_type_str)`` as returned by
                     ``load_dataset()``, or a 2-tuple ``(smiles, labels)`` when
                     ``task_type`` is specified explicitly.
        task_type:   ``"classification"``, ``"regression"``, or ``"auto"`` (use
                     the third element of the tuple, or detect from label values).
        fp_type:     Fingerprint type(s).  Pass a single string or a list:

                     * single string → returns one result dict
                     * list of strings → returns ``dict[fp_type → result_dict]``

                     Choices: ``"ecfp4"`` (default), ``"maccs"``,
                     ``"rdkit_topo"``, ``"atompair"``.
        max_samples: Max molecules before subsampling (classification / NNDR).
        n_repeats:   Subsampling repeats for variance estimation.
        seed:        Random seed.

    Returns:
        If ``fp_type`` is a str:
            ``dict`` with keys ``task_type``, ``fp_type``, ``metric``,
            ``mean``, ``std``, ``n_valid``.
        If ``fp_type`` is a list:
            ``dict[str, dict]`` keyed by fingerprint name.

    Examples::

        smiles, labels, task = eb.load_dataset("hepato_tox")

        # Single fingerprint (default)
        r = eb.roughness((smiles, labels, task))
        # {"task_type": "classification", "fp_type": "ecfp4",
        #  "metric": "NNDR", "mean": 0.31, "std": 0.02, "n_valid": 450}

        # All four fingerprints
        r = eb.roughness((smiles, labels, task),
                         fp_type=["ecfp4", "maccs", "rdkit_topo", "atompair"])
        # {"ecfp4": {...}, "maccs": {...}, "rdkit_topo": {...}, "atompair": {...}}

        # Local CSV dataset
        smiles, labels, task = eb.load_dataset("my_data.csv")
        r = eb.roughness((smiles, labels, task), task_type="regression")
    """
    # 1. Unpack dataset tuple
    if len(dataset) == 3:
        smiles, labels, detected_task = dataset
        if task_type == "auto":
            task_type = detected_task
    elif len(dataset) == 2:
        smiles, labels = dataset
    else:
        raise ValueError("dataset must be a 2-tuple (smiles, labels) or 3-tuple (smiles, labels, task_type).")

    # 2. Auto-detect if still "auto"
    if task_type == "auto":
        labels_arr = np.asarray(labels, dtype=float)
        unique_vals = set(np.unique(labels_arr[~np.isnan(labels_arr)]).tolist())
        task_type = "classification" if unique_vals.issubset({0.0, 1.0}) else "regression"

    labels = np.asarray(labels, dtype=float)

    # 3. Compute for a single fingerprint type
    def _compute_one(fp):
        if task_type == "classification":
            mean, std, n = compute_nndr(smiles, labels, fp, max_samples, n_repeats, seed)
            metric = "NNDR"
        else:
            mean, std, n = compute_sali(smiles, labels, fp, n_repeats=n_repeats, seed=seed)
            metric = "SALI"
        return {"task_type": task_type, "fp_type": fp, "metric": metric,
                "mean": mean, "std": std, "n_valid": n}

    # 4. Dispatch: single string or list
    if isinstance(fp_type, str):
        return _compute_one(fp_type)
    else:
        return {fp: _compute_one(fp) for fp in fp_type}


def compute_roughness(smiles, labels, task_type="auto", fp_type="ecfp4",
                      max_samples=2000, n_repeats=10, seed=42):
    """
    Compute landscape roughness from separate smiles and labels arrays.

    Convenience wrapper around :func:`roughness` for when you already have
    smiles and labels as separate variables (e.g. after custom preprocessing).

    Args:
        smiles:      List of SMILES strings.
        labels:      Array-like of labels.
        task_type:   ``"classification"``, ``"regression"``, or ``"auto"``.
        fp_type:     Fingerprint type(s) — str or list of str (see :func:`roughness`).
        max_samples: Max molecules before subsampling.
        n_repeats:   Subsampling repeats.
        seed:        Random seed.

    Returns:
        Same as :func:`roughness`.

    Examples::

        r = eb.compute_roughness(smiles, labels, task_type="classification")
        r = eb.compute_roughness(smiles, labels, fp_type=["ecfp4", "maccs"])
    """
    return roughness((smiles, labels), task_type=task_type, fp_type=fp_type,
                     max_samples=max_samples, n_repeats=n_repeats, seed=seed)
