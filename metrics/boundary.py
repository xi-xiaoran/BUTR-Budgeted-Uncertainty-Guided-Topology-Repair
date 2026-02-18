from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import cv2

def _surface(mask: np.ndarray) -> np.ndarray:
    er = binary_erosion(mask, iterations=1)
    return mask ^ er

def hd95(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0: return 0.0
    if pred.sum()==0 or gt.sum()==0: return float("nan")
    sp=_surface(pred); sg=_surface(gt)
    dtg = distance_transform_edt(~sg)
    dtp = distance_transform_edt(~sp)
    d = np.concatenate([dtg[sp], dtp[sg]], 0)
    return float(np.percentile(d,95)) if d.size else 0.0

def asd_assd(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0: return 0.0, 0.0
    if pred.sum()==0 or gt.sum()==0: return float("nan"), float("nan")
    sp=_surface(pred); sg=_surface(gt)
    dtg = distance_transform_edt(~sg)
    dtp = distance_transform_edt(~sp)
    d1 = dtg[sp]; d2 = dtp[sg]
    if d1.size==0 or d2.size==0: return 0.0, 0.0
    asd1 = float(d1.mean()); asd2 = float(d2.mean())
    return asd1, float((asd1+asd2)/2)

def _boundary_map(mask: np.ndarray):
    ker = np.ones((3,3), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), ker, iterations=1)
    er  = cv2.erode(mask.astype(np.uint8), ker, iterations=1)
    return (dil-er) > 0

def bf1(pred: np.ndarray, gt: np.ndarray, tol:int=1):
    pred=pred.astype(bool); gt=gt.astype(bool)
    bp=_boundary_map(pred); bg=_boundary_map(gt)
    if bp.sum()==0 and bg.sum()==0: return 1.0
    if bp.sum()==0 or bg.sum()==0: return 0.0
    dtg=distance_transform_edt(~bg)
    dtp=distance_transform_edt(~bp)
    mp = (dtg[bp] <= tol).sum()
    mg = (dtp[bg] <= tol).sum()
    prec = mp/(bp.sum()+1e-6); rec = mg/(bg.sum()+1e-6)
    return float(2*prec*rec/(prec+rec+1e-6))

def bf1_multi(pred, gt, tols=(1,2,3)):
    return {f"bf1@{t}": bf1(pred, gt, t) for t in tols}
