import torch
import torch.nn.functional as F


def focal_tversky_loss(logits: torch.Tensor,
                       target: torch.Tensor,
                       alpha: float = 0.3,
                       beta: float = 0.7,
                       gamma: float = 0.75,
                       mask: torch.Tensor | None = None,
                       eps: float = 1e-6) -> torch.Tensor:
    """Focal-Tversky loss for binary segmentation.

    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    mask:   optional [B,1,H,W] in {0,1}, restricts the region for TP/FP/FN stats.
    """
    p = torch.sigmoid(logits)
    y = target.float()

    if mask is not None:
        m = mask.float()
        p = p * m
        y = y * m

    dims = (2, 3)
    tp = (p * y).sum(dims)
    fp = (p * (1.0 - y)).sum(dims)
    fn = ((1.0 - p) * y).sum(dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = torch.pow((1.0 - tversky), gamma)
    return loss.mean()