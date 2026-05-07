"""
Conformal prediction and uncertainty calibration for molecular property prediction.

Provides split-conformal prediction intervals (regression) and prediction sets
(classification), plus Expected Calibration Error (ECE) metrics.

Quick start::

    import numpy as np
    import envmolbench as eb

    # Classification — pass predicted probabilities
    result = eb.conformal_prediction(y_true, y_prob, task_type="classification")
    ece    = eb.compute_ece(y_true, y_prob, task_type="classification")

    # Regression — pass point predictions (and optionally predictive std)
    result = eb.conformal_prediction(y_true, y_pred, y_std=y_std, task_type="regression")
    ece    = eb.compute_ece(y_true, y_pred, y_std=y_std, task_type="regression")

    print(result)
    # {"task_type": "regression", "nominal_level": 0.9,
    #  "coverage_mean": 0.912, "coverage_std": 0.018,
    #  "calibration_factor": 1.03, "ece": 0.041, "n_samples": 200}
"""

from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

__all__ = [
    "conformal_prediction",
    "compute_ece",
    "conformal_regression",
    "conformal_classification",
    "compute_ece_regression",
    "compute_ece_classification",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_cal_eval(n: int, cal_frac: float = 0.5, seed: int = 42):
    """Random 50/50 calibration / evaluation split."""
    perm = np.random.RandomState(seed).permutation(n)
    cal_end = int(n * cal_frac)
    return perm[:cal_end], perm[cal_end:]


def _detect_task_type(y_true: np.ndarray) -> str:
    valid = y_true[~np.isnan(y_true)]
    unique_vals = set(np.unique(valid).tolist())
    if unique_vals.issubset({0.0, 1.0}) and len(unique_vals) <= 2:
        return "classification"
    return "regression"


# ── Core split-conformal functions ────────────────────────────────────────────

def conformal_regression(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    nominal_level: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Split-conformal prediction interval for regression.

    Randomly splits the data 50/50 into calibration and evaluation sets,
    computes conformity scores on the calibration set, and measures
    interval coverage on the evaluation set.

    Args:
        y_true:        Ground-truth continuous labels.
        y_mean:        Point predictions (ensemble mean or single model).
        y_std:         Predictive uncertainty (ensemble std).
        nominal_level: Target coverage level, e.g. 0.90.
        seed:          Random seed for the 50/50 split.

    Returns:
        dict with keys:
            ``coverage``           — empirical coverage on the evaluation set
            ``calibration_factor`` — q / z (Gaussian calibration ratio)
            ``nominal_level``      — requested coverage level
            ``lower_bounds``       — lower interval bounds on evaluation set
            ``upper_bounds``       — upper interval bounds on evaluation set
    """
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std  = np.asarray(y_std, dtype=float)

    cal_idx, eval_idx = _split_cal_eval(len(y_true), seed=seed)
    cal_scores = (
        np.abs(y_true[cal_idx] - y_mean[cal_idx])
        / np.maximum(y_std[cal_idx], 1e-8)
    )
    n_cal = len(cal_scores)
    q = np.quantile(
        cal_scores,
        min(np.ceil((n_cal + 1) * nominal_level) / n_cal, 1.0),
    )
    std_eval = np.maximum(y_std[eval_idx], 1e-8)
    lower = y_mean[eval_idx] - q * std_eval
    upper = y_mean[eval_idx] + q * std_eval
    coverage = float(
        ((y_true[eval_idx] >= lower) & (y_true[eval_idx] <= upper)).mean()
    )
    z = scipy_stats.norm.ppf((1 + nominal_level) / 2)
    return {
        "coverage": coverage,
        "calibration_factor": float(q / z) if z > 0 else float(q),
        "nominal_level": nominal_level,
        "lower_bounds": lower,
        "upper_bounds": upper,
    }


def conformal_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    nominal_level: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Split-conformal prediction set for binary classification.

    Args:
        y_true:        Ground-truth binary labels (0/1).
        y_prob:        Predicted probability for the positive class.
        nominal_level: Target coverage level.
        seed:          Random seed.

    Returns:
        dict with keys:
            ``coverage``      — empirical coverage on the evaluation set
            ``threshold``     — conformal score threshold q
            ``nominal_level`` — requested level
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    cal_idx, eval_idx = _split_cal_eval(len(y_true), seed=seed)
    cal_scores = np.where(
        y_true[cal_idx].astype(int) == 1,
        1 - y_prob[cal_idx],
        y_prob[cal_idx],
    )
    n_cal = len(cal_scores)
    q = np.quantile(
        cal_scores,
        min(np.ceil((n_cal + 1) * nominal_level) / n_cal, 1.0),
    )
    eval_scores = np.where(
        y_true[eval_idx].astype(int) == 1,
        1 - y_prob[eval_idx],
        y_prob[eval_idx],
    )
    coverage = float((eval_scores <= q).mean())
    return {"coverage": coverage, "threshold": float(q), "nominal_level": nominal_level}


# ── ECE functions ─────────────────────────────────────────────────────────────

def compute_ece_regression(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_levels: int = 9,
) -> float:
    """
    Expected Calibration Error for regression under a Gaussian predictive model.

    Computes ECE = mean |empirical_coverage(level) - level| over n_levels
    uniformly spaced from 0.1 to 0.9.

    Args:
        y_true:   Ground-truth continuous labels.
        y_mean:   Predicted means.
        y_std:    Predicted standard deviations.
        n_levels: Number of confidence levels to average over.

    Returns:
        Scalar ECE value (lower is better calibrated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std  = np.asarray(y_std, dtype=float)

    ece = 0.0
    for level in np.linspace(0.1, 0.9, n_levels):
        z = scipy_stats.norm.ppf((1 + level) / 2)
        within = ((y_true >= y_mean - z * y_std) & (y_true <= y_mean + z * y_std)).mean()
        ece += abs(within - level)
    return float(ece / n_levels)


def compute_ece_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for binary classification.

    Uses equal-width binning of confidence scores from 0.5 to 1.0.

    Args:
        y_true:  Ground-truth binary labels (0/1).
        y_prob:  Predicted positive-class probability.
        n_bins:  Number of confidence bins.

    Returns:
        Scalar ECE value (lower is better calibrated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    conf  = np.maximum(y_prob, 1 - y_prob)
    preds = (y_prob >= 0.5).astype(int)
    accs  = (preds == y_true.astype(int)).astype(float)
    bounds = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        # Include lower bound for the first bin (conf == 0.5)
        mask = (conf >= bounds[i]) & (conf <= bounds[i + 1]) if i == 0 else \
               (conf >  bounds[i]) & (conf <= bounds[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(accs[mask].mean() - conf[mask].mean())
    return float(ece)


# ── Unified wrappers ──────────────────────────────────────────────────────────

def conformal_prediction(
    y_true,
    y_pred,
    y_std=None,
    task_type: str = "auto",
    nominal_level: float = 0.90,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict:
    """
    Unified conformal prediction for regression or classification.

    Runs ``n_repeats`` random 50/50 calibration/evaluation splits and averages
    coverage to reduce variance.

    Args:
        y_true:        Ground-truth labels (continuous for regression, 0/1 for
                       classification).
        y_pred:        Predictions — continuous values for regression, predicted
                       positive-class probability for classification.
        y_std:         Predictive std for regression (optional).  If ``None``,
                       a constant value equal to the residual std is used.
                       Ignored for classification.
        task_type:     ``"regression"``, ``"classification"``, or ``"auto"``
                       (detects from ``y_true``: values in {0, 1} → classification).
        nominal_level: Target coverage level (default 0.90).
        n_repeats:     Number of random splits to average over (default 10).
        seed:          Base random seed; each repeat uses ``seed + i``.

    Returns:
        dict with keys:
            ``task_type``          — resolved task type used
            ``nominal_level``      — requested coverage level
            ``coverage_mean``      — mean empirical coverage across repeats
            ``coverage_std``       — std of coverage across repeats
            ``calibration_factor`` — mean q/z ratio (regression only, else None)
            ``ece``                — Expected Calibration Error
            ``n_samples``          — number of samples

    Examples::

        # Classification
        result = eb.conformal_prediction(y_true, y_prob, task_type="classification")

        # Regression with ensemble uncertainty
        result = eb.conformal_prediction(y_true, y_pred, y_std=y_std)

        # Auto-detect + multiple repeats
        result = eb.conformal_prediction(y_true, y_pred, n_repeats=20)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if task_type == "auto":
        task_type = _detect_task_type(y_true)

    if task_type == "regression":
        if y_std is None:
            residuals = np.abs(y_true - y_pred)
            y_std = np.full_like(y_pred, np.std(residuals))
        else:
            y_std = np.asarray(y_std, dtype=float)

    coverages = []
    cal_factors = []

    for i in range(n_repeats):
        if task_type == "regression":
            res = conformal_regression(y_true, y_pred, y_std, nominal_level, seed + i)
            coverages.append(res["coverage"])
            cal_factors.append(res["calibration_factor"])
        else:
            res = conformal_classification(y_true, y_pred, nominal_level, seed + i)
            coverages.append(res["coverage"])

    if task_type == "regression":
        ece = compute_ece_regression(y_true, y_pred, y_std)
    else:
        ece = compute_ece_classification(y_true, y_pred)

    return {
        "task_type": task_type,
        "nominal_level": nominal_level,
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "calibration_factor": float(np.mean(cal_factors)) if cal_factors else None,
        "ece": float(ece),
        "n_samples": int(len(y_true)),
    }


def compute_ece(
    y_true,
    y_pred,
    y_std=None,
    task_type: str = "auto",
) -> float:
    """
    Unified Expected Calibration Error for regression or classification.

    Args:
        y_true:    Ground-truth labels.
        y_pred:    Predictions (continuous or positive-class probability).
        y_std:     Predictive std (regression only). If ``None``, uses residual std.
        task_type: ``"regression"``, ``"classification"``, or ``"auto"``.

    Returns:
        Scalar ECE value (lower = better calibrated).

    Examples::

        ece = eb.compute_ece(y_true, y_prob, task_type="classification")
        ece = eb.compute_ece(y_true, y_pred, y_std=y_std, task_type="regression")
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if task_type == "auto":
        task_type = _detect_task_type(y_true)

    if task_type == "regression":
        if y_std is None:
            residuals = np.abs(y_true - y_pred)
            y_std = np.full_like(y_pred, np.std(residuals))
        else:
            y_std = np.asarray(y_std, dtype=float)
        return compute_ece_regression(y_true, y_pred, y_std)
    else:
        return compute_ece_classification(y_true, y_pred)
