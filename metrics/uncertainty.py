from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def ece(prob: np.ndarray, gt: np.ndarray, n_bins=15):
    y = gt.astype(np.uint8).ravel()
    p = prob.astype(np.float32).ravel()
    if len(np.unique(y))<2: return np.nan
    bins = np.linspace(0,1,n_bins+1)
    e=0.0
    for i in range(n_bins):
        m=(p>=bins[i]) & (p<bins[i+1])
        if m.sum()==0: continue
        acc=y[m].mean(); conf=p[m].mean()
        e += (m.sum()/len(p))*abs(acc-conf)
    return float(e)

def error_alignment_auc(uncert: np.ndarray, err_mask: np.ndarray):
    y = err_mask.astype(np.uint8).ravel()
    s = uncert.astype(np.float32).ravel()
    if len(np.unique(y))<2: return np.nan, np.nan
    try:
        return float(roc_auc_score(y,s)), float(average_precision_score(y,s))
    except Exception:
        return np.nan, np.nan

def overlap_iou_recall_budget(uncert: np.ndarray, viol: np.ndarray, budget=0.1):
    u = uncert.astype(np.float32)
    v = viol.astype(bool)
    k = int(budget*u.size)
    if k<=0: return np.nan, np.nan
    thr = np.partition(u.ravel(), -k)[-k]
    roi = u >= thr
    inter = (roi & v).sum()
    union = (roi | v).sum()
    iou = inter/(union+1e-6) if union>0 else np.nan
    rec = inter/(v.sum()+1e-6) if v.sum()>0 else np.nan
    return float(iou), float(rec)
