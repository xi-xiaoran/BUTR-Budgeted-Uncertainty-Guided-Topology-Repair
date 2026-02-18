from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = (pred & gt).sum()
    dice = (2*inter + eps)/((pred.sum()+gt.sum())+eps)
    iou  = (inter + eps)/(((pred|gt).sum())+eps)
    return float(dice), float(iou)

def sens_spec(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = (pred & gt).sum()
    tn = ((~pred) & (~gt)).sum()
    fp = (pred & (~gt)).sum()
    fn = ((~pred) & gt).sum()
    sens = (tp+eps)/(tp+fn+eps)
    spec = (tn+eps)/(tn+fp+eps)
    return float(sens), float(spec)

def auc_aupr(prob: np.ndarray, gt: np.ndarray):
    y = gt.astype(np.uint8).ravel()
    s = prob.astype(np.float32).ravel()
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        return float(roc_auc_score(y,s)), float(average_precision_score(y,s))
    except Exception:
        return np.nan, np.nan
