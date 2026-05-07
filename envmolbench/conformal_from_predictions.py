"""
Deprecated: this module has been replaced by envmolbench.conformal.

Use the new library API instead::

    import envmolbench as eb

    # Unified wrapper (recommended)
    result = eb.conformal_prediction(y_true, y_pred, task_type="classification")
    ece    = eb.compute_ece(y_true, y_pred, task_type="regression")

    # Low-level functions
    from envmolbench.conformal import (
        conformal_regression,
        conformal_classification,
        compute_ece_regression,
        compute_ece_classification,
    )
"""
import warnings

warnings.warn(
    "envmolbench.conformal_from_predictions is deprecated and will be removed in a future version. "
    "Use envmolbench.conformal (or import via `import envmolbench as eb`) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .conformal import (
    conformal_regression,
    conformal_classification,
    compute_ece_regression,
    compute_ece_classification,
)

__all__ = [
    "conformal_regression",
    "conformal_classification",
    "compute_ece_regression",
    "compute_ece_classification",
]
