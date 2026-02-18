from __future__ import annotations
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border

def topo_stats(mask: np.ndarray):
    m = mask.astype(bool)
    fg_ratio = float(m.mean())
    MIN_FG = 1e-4  # avoid treating empty prediction as topology pass
    lab = measure.label(m, connectivity=2)
    n_cc = int(lab.max())
    main_ratio = 0.0
    if n_cc>0:
        areas = np.bincount(lab.ravel()); areas[0]=0
        main_ratio = float(areas.max()/(areas.sum()+1e-6))
    inv = (~m).astype(np.uint8)
    inv = clear_border(inv)
    holes = int(measure.label(inv, connectivity=1).max())
    euler = float(measure.euler_number(m, connectivity=2))
    topo_pass = float((fg_ratio >= MIN_FG) and (n_cc<=1) and (holes==0))
    return {"n_cc": n_cc, "main_cc_ratio": main_ratio, "holes": holes, "euler": euler, "fg_ratio": fg_ratio, "topo_pass": topo_pass}

def betti_matching_error(pred_stats: dict, gt_stats: dict):
    b0p, b1p = pred_stats.get("n_cc", np.nan), pred_stats.get("holes", np.nan)
    b0g, b1g = gt_stats.get("n_cc", np.nan), gt_stats.get("holes", np.nan)
    if np.isnan(b0p) or np.isnan(b0g): return np.nan
    return float(abs(b0p-b0g) + abs(b1p-b1g))