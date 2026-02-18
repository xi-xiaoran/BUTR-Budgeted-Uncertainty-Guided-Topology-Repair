from __future__ import annotations

import numpy as np
from skimage import measure, morphology
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border

# Treat (nearly) empty predictions as NOT passing the topology certificate.
# This prevents trivial "all background" predictions from getting topo_pass=1.
MIN_FG = 1e-4


def _holes(mask: np.ndarray) -> int:
    """Count interior holes by labeling background components after clearing border."""
    inv = (~mask).astype(np.uint8)
    inv = clear_border(inv)
    lab = measure.label(inv, connectivity=1)
    return int(lab.max())


def certify_mask(mask: np.ndarray, min_area: int = 10, min_fg: float = MIN_FG):
    """Topology certificate for a binary mask.

    Returns a dict with:
      - topo_pass: bool
      - n_cc: number of connected components
      - main_cc_ratio: area(main_cc)/area(all_fg)
      - holes: number of holes
      - viol_map: uint8 map in {0,255} indicating violating pixels/regions
      - fg_ratio: foreground pixel ratio (after min_area filtering)
    """
    m = mask.astype(bool)
    if min_area and min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))

    fg_ratio = float(m.mean()) if m.size > 0 else 0.0

    # Near-empty masks are considered invalid (avoid trivial passing).
    if fg_ratio < float(min_fg):
        viol = np.ones_like(m, np.uint8) * 255
        return {
            "topo_pass": False,
            "n_cc": 0,
            "main_cc_ratio": 0.0,
            "holes": 0,
            "viol_map": viol,
            "fg_ratio": fg_ratio,
        }

    lab = measure.label(m.astype(np.uint8), connectivity=1)
    n_cc = int(lab.max())

    if n_cc == 0:
        # Should be rare given fg_ratio>=min_fg, but keep safe.
        viol = np.ones_like(m, np.uint8) * 255
        return {
            "topo_pass": False,
            "n_cc": 0,
            "main_cc_ratio": 0.0,
            "holes": 0,
            "viol_map": viol,
            "fg_ratio": fg_ratio,
        }

    areas = np.bincount(lab.ravel())
    if areas.size > 0:
        areas[0] = 0
    main = int(areas.argmax())
    main_ratio = float(areas[main] / (areas.sum() + 1e-6))

    # Violations: pixels not belonging to the main CC
    # viol = (lab != main).astype(np.uint8) * 255# Violations: foreground pixels not belonging to the main CC (exclude background)
    viol = ((lab > 0) & (lab != main)).astype(np.uint8) * 255


    holes = _holes(m)
    if holes > 0:
        # Mark hole regions as violations as well
        filled = morphology.remove_small_holes(m, area_threshold=64)
        hole_region = (filled & (~m)).astype(np.uint8) * 255
        viol = np.maximum(viol, hole_region)

    topo_pass = (n_cc <= 1) and (holes == 0)

    return {
        "topo_pass": bool(topo_pass),
        "n_cc": int(n_cc),
        "main_cc_ratio": float(main_ratio),
        "holes": int(holes),
        "viol_map": viol,
        "fg_ratio": fg_ratio,
    }
