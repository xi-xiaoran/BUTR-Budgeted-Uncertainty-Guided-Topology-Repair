from __future__ import annotations
import numpy as np
from skimage.morphology import skeletonize

def skeleton_dice(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    pred=pred.astype(bool); gt=gt.astype(bool)
    sp=skeletonize(pred); sg=skeletonize(gt)
    union = sp.sum()+sg.sum()
    if union==0: return 1.0
    inter=(sp&sg).sum()
    return float((2*inter+eps)/(union+eps))

def cldice_metric(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    pred=pred.astype(bool); gt=gt.astype(bool)
    sp=skeletonize(pred); sg=skeletonize(gt)
    tprec = (sp & gt).sum()/(sp.sum()+eps)
    tsens = (sg & pred).sum()/(sg.sum()+eps)
    return float(2*tprec*tsens/(tprec+tsens+eps))
