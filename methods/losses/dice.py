import torch
import torch.nn.functional as F

def _dice_from_logits(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p*target).sum(dim=(2,3))
    union = p.sum(dim=(2,3)) + target.sum(dim=(2,3))
    d = (2*inter + eps)/(union + eps)
    return 1 - d.mean()

def ce_dice_loss(logits, target, ce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dl = _dice_from_logits(logits, target)
    return ce_weight*bce + dice_weight*dl
