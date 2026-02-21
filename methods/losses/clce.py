from __future__ import annotations

import torch
import torch.nn.functional as F

from .cldice import _soft_skel as soft_skel


def soft_clce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iters: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Centerline-Cross Entropy (clCE) loss for binary segmentation.

    Based on:
      Acebes et al., "The clCE loss for vessel-like structure segmentation", MICCAI 2024.

    Args:
        logits: (B, 1, H, W) raw logits for foreground.
        target: (B, 1, H, W) or (B, H, W) binary mask in {0,1}.
        iters: soft skeleton iterations.
        eps: numerical stability.

    Returns:
        scalar loss tensor.
    """
    if logits.dim() != 4 or logits.size(1) != 1:
        raise ValueError(f"logits must be (B,1,H,W), got {tuple(logits.shape)}")

    if target.dim() == 3:
        target_f = target.unsqueeze(1).float()
    elif target.dim() == 4:
        target_f = target.float()
        if target_f.size(1) != 1:
            # If one-hot (B,2,H,W), pick foreground channel
            if target_f.size(1) == 2:
                target_f = target_f[:, 1:2]
            else:
                raise ValueError(f"target must have 1 channel, got {target_f.size(1)}")
    else:
        raise ValueError(f"target must be (B,H,W) or (B,1,H,W), got {tuple(target.shape)}")

    # Per-pixel CE (binary) (B,1,H,W)
    l_ce = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none")

    # Soft skeletons (foreground)
    prob = torch.sigmoid(logits)
    sk_true = soft_skel(target_f, iters=iters)
    sk_pred = soft_skel(prob, iters=iters)

    # Paper implementation uses mean of weighted CE terms (equivalent to a weighted average up to a constant).
    ce_tprec = (l_ce * sk_true).mean()
    ce_trecall = (l_ce * sk_pred).mean()
    return ce_tprec + ce_trecall
