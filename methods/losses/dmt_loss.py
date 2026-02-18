import torch
import torch.nn.functional as F

def _ste(prob, thr=0.5):
    hard = (prob > thr).float()
    return hard + (prob - prob.detach())

def _neighbors8(x):
    k = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    return F.conv2d(x, k, padding=1)

def dmt_loss_proxy(logits, target, thr=0.5):
    prob = torch.sigmoid(logits)
    hard = _ste(prob, thr)
    gt = (target > 0.5).float()
    nb = _neighbors8(hard); nb_g = _neighbors8(gt)
    end = (hard * (nb==1).float()).sum(dim=(2,3))
    br  = (hard * (nb>=3).float()).sum(dim=(2,3))
    end_g = (gt * (nb_g==1).float()).sum(dim=(2,3))
    br_g  = (gt * (nb_g>=3).float()).sum(dim=(2,3))
    return (end-end_g).abs().mean() + 0.5*(br-br_g).abs().mean()
