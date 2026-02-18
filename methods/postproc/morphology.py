from __future__ import annotations
import numpy as np
import cv2
from skimage.morphology import remove_small_objects

def morph_postproc(prob: np.ndarray, thr=0.5, k_close=3, k_open=3, min_cc=20):
    m = (prob > thr).astype(np.uint8)
    if k_close>0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close,k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    if k_open>0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    if min_cc>0:
        m = remove_small_objects(m.astype(bool), min_size=min_cc).astype(np.uint8)
    return m
