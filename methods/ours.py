import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from methods.losses import edl_uncertainty_from_evidence


# ---------------------- small utils ----------------------
def _zero_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _kaiming_init(m: nn.Module):
    """Kaiming init for conv trunk; keeps gradients alive under ReLU."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _dilate_binary(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Binary-ish dilation for NCHW mask in {0,1}."""
    if k <= 1:
        return x
    p = k // 2
    return (F.max_pool2d(x, kernel_size=k, stride=1, padding=p) > 0).float()


def _roi_topk(unc: torch.Tensor, topk: float = 0.10) -> torch.Tensor:
    """Per-image ROI selecting top-k fraction highest uncertainty. Returns {0,1} mask."""
    topk = float(topk)
    topk = min(max(topk, 1e-4), 1.0)
    B = unc.shape[0]
    flat = unc.view(B, -1)
    # pick threshold as (1-topk) quantile
    q = 1.0 - topk
    try:
        thr = torch.quantile(flat, q, dim=1, keepdim=True)  # [B,1]
    except Exception:
        # fallback: sort
        k = max(1, int(flat.shape[1] * topk))
        thr, _ = torch.kthvalue(flat, flat.shape[1] - k + 1, dim=1, keepdim=True)
    thr = thr.view(B, 1, 1, 1)
    return (unc >= thr).float()


def _safe_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


# ---------------------- repair heads ----------------------
class RepairHeadV1(nn.Module):
    """V1: lightweight repair head (delta-logit only).

    Important: DO NOT zero-init all conv layers with ReLU, otherwise gradients die
    (ReLU'(0)=0) and the head collapses to bias-only behavior, making V1/V2 look
    identical. We instead Kaiming-init the trunk and small-init the output.

    Output: delta_logit [B,1,H,W]
    """

    def __init__(self, in_ch=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 1, 1)

        # init: make trunk learnable, output near 0 (no-op at start)
        self.net.apply(_kaiming_init)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.net(x))


class RepairHeadV2(nn.Module):
    """V2: stronger repair head with a learned gate.

    Output: [B,2,H,W] where:
      - out[:,0:1] = delta_logit
      - out[:,1:2] = gate_logit  (sigmoid -> (0,1))
    """

    def __init__(self, in_ch=6, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 2, 1)

        # init trunk normally; output small; gate starts "closed"
        self.net.apply(_kaiming_init)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)
        # gate bias negative => sigmoid(gate) small at init, delta bias stays 0
        with torch.no_grad():
            self.out.bias[1].fill_(-3.0)

    def forward(self, x):
        return self.out(self.net(x))

# ---------------------- core ----------------------
@torch.no_grad()
def _base_forward(base_model, head_mode: str, img: torch.Tensor):
    out = base_model(img)
    if head_mode == "standard":
        prob = torch.sigmoid(out)
        unc = torch.zeros_like(prob)
    else:
        # EDL evidence head
        prob, unc = edl_uncertainty_from_evidence(out)
    return prob, unc
# ---------------------- optional certificate ----------------------
def _make_roi(base_prob: torch.Tensor,
              uncert: torch.Tensor,
              viol: torch.Tensor,
              tau_unc: float = 0.5,
              thr: float = 0.5) -> torch.Tensor:
    """Build a binary ROI mask for repair.

    Heuristic (robust + simple):
      - include all topology-violation pixels (viol > 0)
      - include pixels whose uncertainty exceeds tau_unc
      - if ROI becomes empty for a sample, fall back to the single most-uncertain pixel

    Shapes:
      base_prob: [B,1,H,W] (only used for fallback)
      uncert   : [B,1,H,W] or None
      viol     : [B,1,H,W] or None
    Returns:
      roi: float mask [B,1,H,W] in {0,1}
    """
    # normalize inputs
    if base_prob is None:
        raise ValueError("base_prob is required")

    B, _, H, W = base_prob.shape
    device = base_prob.device

    roi_bool = torch.zeros((B, 1, H, W), device=device, dtype=torch.bool)

    if uncert is not None:
        if uncert.dim() == 3:
            uncert = uncert.unsqueeze(1)
        if uncert.shape[1] != 1:
            uncert = uncert[:, -1:, ...]
        # Note: uncert expected in [0,1]
        roi_bool |= (uncert >= float(tau_unc))

    if viol is not None:
        if viol.dim() == 3:
            viol = viol.unsqueeze(1)
        if viol.shape[1] != 1:
            viol = viol[:, -1:, ...]
        roi_bool |= (viol > 0)

    roi = roi_bool.float()

    # fallback: make sure ROI is not empty per-sample
    roi_flat = roi.reshape(B, -1)
    empty = roi_flat.sum(dim=1) < 1
    if empty.any():
        if uncert is not None:
            u_flat = uncert.reshape(B, -1)
            argmax = u_flat.argmax(dim=1)  # [B]
            # set one pixel for empty samples
            roi_flat[empty] = 0.0
            roi_flat[empty, argmax[empty]] = 1.0
            roi = roi_flat.reshape(B, 1, H, W)
        else:
            # fall back to near-threshold band (if no uncertainty provided)
            roi = ((base_prob - float(thr)).abs() <= 0.05).float()

    return roi



def _certify_batch(prob: torch.Tensor, thr: float = 0.5):
    """Return (viol_map, topo_info) for a batch.

    Return a binary violation map computed by a fast topology/structure certifier.

    The certifier is enabled by default. To disable it (e.g., for ablations), set:
        TOPO_CERTIFY=0
    """
    # enforce single-channel foreground prob
    if prob.dim() == 4 and prob.shape[1] > 1:
        prob = prob[:, -1:, ...]
    B, _, H, W = prob.shape
    viol = torch.zeros((B, 1, H, W), device=prob.device, dtype=prob.dtype)
    topo = {"topo_pass_rate": 1.0, "used": False}

    if os.environ.get("TOPO_CERTIFY", "1") not in ("1", "true", "True"):
        return viol, topo

    # best-effort certificate (CPU, can be slow). Any failure falls back to zeros.
    try:
        import numpy as np
        from methods.certify.certificate import certify_mask

        viol_list = []
        pass_list = []
        for b in range(B):
            m = (prob[b, 0].detach().float().cpu().numpy() > thr).astype(np.uint8)
            res = certify_mask(m)
            vm = res.get("viol_map", None)
            if vm is None:
                vm = np.zeros_like(m, dtype=np.uint8)
            viol_list.append(torch.from_numpy((vm > 0).astype(np.float32))[None, None, ...])
            pass_list.append(float(res.get("is_topo_pass", 1)))

        viol = torch.cat(viol_list, dim=0).to(device=prob.device, dtype=prob.dtype)
        topo = {"topo_pass_rate": float(sum(pass_list) / max(1, len(pass_list))), "used": True}
        return viol, topo
    except Exception as e:
        topo = {"topo_pass_rate": 1.0, "used": False, "err": str(e)[:120]}
        return viol, topo




def run_ours_repair(base_model, head_mode: str, repair_head, img,
                    tau_unc=0.5, thr=0.5, iters=1,
                    roi_topk=None,
                    roi_override=None,
                    roi_mode: str = "union",
                    uncert_override=None,
                    viol_override=None):
    """
    BUTR (Budgeted Uncertainty-guided Topology Repair) forward.

    Key idea: only edit within a high-risk ROI. By default (roi_mode="union"),
    ROI is built from (uncertainty >= tau_unc) UNION (topology/rule violation map > 0).

    Args:
      roi_override:
        If provided (training), bypass ROI construction and use this binary mask.
      roi_mode:
        {"unc","viol","union"} to ablate ROI construction at inference time.
      uncert_override / viol_override:
        If provided, replace the corresponding maps used by ROI construction and
        as repair-head inputs. Passing `viol_override` also bypasses internal certify.

    Returns:
      refined_prob, uncert, aux(dict)
    """
    roi_mode = (roi_mode or "union").lower().strip()
    if roi_mode not in ("unc", "viol", "union"):
        raise ValueError(f"Unknown roi_mode={roi_mode}. Use one of: unc, viol, union.")

    out = base_model(img)
    if head_mode == "standard":
        base_prob = torch.sigmoid(out)
        uncert = torch.zeros_like(base_prob)
    else:
        base_prob, uncert = edl_uncertainty_from_evidence(out)

    # --- keep FG channel only (some backbones output 2ch logits/probs) ---
    if base_prob.dim() == 4 and base_prob.shape[1] > 1:
        base_prob = base_prob[:, -1:, ...]
    if uncert is None:
        uncert = torch.zeros_like(base_prob)
    if uncert.dim() == 4 and uncert.shape[1] > 1:
        uncert = uncert[:, -1:, ...]

    # --- match spatial size to input image (robust to odd-size down/up sampling) ---
    H, W = img.shape[-2], img.shape[-1]

    def _resize_to_img(t, mode: str):
        if t is None:
            return None
        if t.dim() == 3:
            t = t.unsqueeze(1)
        if t.shape[1] != 1:
            t = t[:, -1:, ...]
        if t.shape[-2:] == (H, W):
            return t
        if mode == "nearest":
            return F.interpolate(t, size=(H, W), mode=mode)
        return F.interpolate(t, size=(H, W), mode=mode, align_corners=False)

    base_prob = _resize_to_img(base_prob, "bilinear")
    uncert = _resize_to_img(uncert, "bilinear")

    # --- optional overrides ---
    if uncert_override is not None:
        u = uncert_override.to(img.device)
        uncert = _resize_to_img(u, "bilinear")
        if uncert is None:
            uncert = torch.zeros_like(base_prob)

    # violation map: either provided (preferred for ablations) or optional internal certify
    topo = {"topo_pass_rate": 1.0, "used": False}
    if viol_override is not None:
        viol = _resize_to_img(viol_override.to(img.device), "nearest")
        if viol is None:
            viol = torch.zeros_like(base_prob)
        topo = {"topo_pass_rate": float("nan"), "used": "override"}
    else:
        viol, topo = _certify_batch(base_prob, thr=thr)
        viol = _resize_to_img(viol.to(img.device), "nearest")
        if viol is None:
            viol = torch.zeros_like(base_prob)

    # --- build maps used by the repair head & ROI ---
    if roi_mode == "unc":
        uncert_used = uncert
        viol_used = torch.zeros_like(viol)
    elif roi_mode == "viol":
        uncert_used = torch.zeros_like(uncert)
        viol_used = viol
    else:  # union
        uncert_used = uncert
        viol_used = viol

    # --- ROI selection ---
    if roi_override is not None:
        roi = _resize_to_img(roi_override.float().to(img.device), "nearest")
        if roi is None:
            roi = torch.zeros_like(base_prob)
    else:
        roi = _make_roi(base_prob, uncert_used, viol_used, tau_unc=tau_unc)

        # optional top-k uncertainty ROI (ONLY when uncertainty is part of roi_mode)
        if roi_topk is not None and float(roi_topk) < 1.0 and roi_mode in ("unc", "union"):
            B, _, HH, WW = uncert_used.shape
            k = max(1, int(float(roi_topk) * HH * WW))
            flat = uncert_used.view(B, -1)
            thrv = torch.kthvalue(flat, flat.size(1) - k + 1, dim=1).values
            thrv = thrv.view(B, 1, 1, 1)
            topk_mask = (uncert_used >= thrv).float()
            roi = torch.maximum(roi, topk_mask)

    roi = (roi > 0.5).float()

    # --- iterative repair ---
    eps = 1e-4
    cur = base_prob
    for _ in range(int(iters)):
        x = torch.cat([img, cur, uncert_used, viol_used], 1)
        out = repair_head(x)
        if out.shape[1] == 1:
            delta = out
            gate = torch.ones_like(delta)
        else:
            delta = out[:, :1, ...]
            gate = torch.sigmoid(out[:, 1:2, ...])
        p = cur.clamp(eps, 1 - eps)
        base_logit = torch.log(p / (1 - p))
        ref_logit = base_logit + (delta * gate) * roi
        cur = torch.sigmoid(ref_logit)

    return cur, uncert, {
        "roi": roi,
        "roi_mode": roi_mode,
        "viol_map": viol_used,
        "base_prob": base_prob.detach(),
        "topo": topo,
    }
