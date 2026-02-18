from __future__ import annotations
import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes

def violation_correct_postproc(prob: np.ndarray, viol_map: np.ndarray|None, thr=0.5, close_k=5, min_cc=20, fill_hole_area=128):
    m = (prob > thr).astype(np.uint8)
    if close_k>0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k,close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    if min_cc>0:
        m = remove_small_objects(m.astype(bool), min_size=min_cc).astype(np.uint8)
    if fill_hole_area>0:
        m = remove_small_holes(m.astype(bool), area_threshold=fill_hole_area).astype(np.uint8)
    if viol_map is not None and (viol_map>0).any():
        v = (viol_map>0).astype(np.uint8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
        m = np.where(v>0, m2, m)
    return m
