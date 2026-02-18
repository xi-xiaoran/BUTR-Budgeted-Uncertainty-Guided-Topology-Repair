from __future__ import annotations

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# NOTE:
# These are lightweight, differentiable *proxy* losses used as stand-ins
# for topology-related constraints (PH-based TopoLoss, DMT-loss, etc.).
# The goal for this repo stage is: "runs everywhere, stable gradients".
# You can later swap these with a full persistent-homology implementation.
# ------------------------------------------------------------------

def _soft_skel_approx(prob: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Very lightweight 'soft skeleton' approximation for 2D masks.
    prob: (B,1,H,W) in [0,1]
    Returns: (B,1,H,W) approx skeleton-like response.
    """
    x = prob
    for _ in range(int(iters)):
        # Erode-ish by taking local min via negative maxpool
        er = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        # Opening-ish: dilate after erosion
        op = F.max_pool2d(er, kernel_size=3, stride=1, padding=1)
        # "skeleton contribution"
        x = torch.relu(x - op)
        # keep within [0,1]
        x = x.clamp_(0.0, 1.0)
    return x

def topo_loss_ph_proxy(logits: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Proxy for PH-based topology loss.
    Encourages skeleton overlap + penalizes spurious fragments (via soft-skel energy).
    logits: (B,1,H,W) raw logits
    target: (B,1,H,W) {0,1}
    """
    prob = torch.sigmoid(logits)
    tgt = target.float()

    sk_p = _soft_skel_approx(prob, iters=6)
    sk_t = _soft_skel_approx(tgt, iters=6)

    # overlap (maximize), use 1 - Dice on skeletons
    inter = (sk_p * sk_t).sum(dim=(2,3))
    denom = sk_p.sum(dim=(2,3)) + sk_t.sum(dim=(2,3)) + 1e-6
    dice = (2 * inter) / denom
    loss = 1.0 - dice.mean()

    # small penalty on excessive skeleton energy (discourage noisy branches)
    energy = sk_p.mean()
    return weight * (loss + 0.1 * energy)

def dmt_loss_proxy(logits: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Proxy for DMT-style loss (distance-map based topology-ish constraint).
    We approximate by weighting BCE more on boundary/transition regions.
    """
    prob = torch.sigmoid(logits)
    tgt = target.float()

    # boundary emphasis using local gradient magnitude of target (cheap)
    gx = torch.abs(F.pad(tgt, (0,1,0,0))[:,:,:,1:] - F.pad(tgt, (1,0,0,0))[:,:,:,:-1])
    gy = torch.abs(F.pad(tgt, (0,0,0,1))[:,:,1:,:] - F.pad(tgt, (0,0,1,0))[:,:,:-1,:])
    # resize gx/gy to same H,W
    gx = F.pad(gx, (0,0,0,0))
    gy = F.pad(gy, (0,0,0,0))
    bnd = (gx + gy).clamp(0.0, 1.0)

    w = 1.0 + 4.0 * bnd  # emphasize boundaries
    bce = F.binary_cross_entropy(prob, tgt, weight=w, reduction="mean")
    return weight * bce
