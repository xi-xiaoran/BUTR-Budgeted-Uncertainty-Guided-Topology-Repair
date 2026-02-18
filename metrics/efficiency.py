from __future__ import annotations
import numpy as np
def changed_frac(pred: np.ndarray, base_pred: np.ndarray):
    return float((pred.astype(bool) ^ base_pred.astype(bool)).mean())
