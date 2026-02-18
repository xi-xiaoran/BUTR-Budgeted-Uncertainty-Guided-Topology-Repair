from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _get_img_mask(batch, device):
    # Unify batch format:
    #   - dict: {"image": Tensor, "mask": Tensor, ...}
    #   - DSItem: has .image and .mask attributes
    if isinstance(batch, dict):
        img = batch["image"]
        mask = batch["mask"]
    else:
        img = getattr(batch, "image")
        mask = getattr(batch, "mask")
    return img.to(device), mask.to(device)


from backbone import build_backbone
from experiments.common import make_loaders, to_device
from methods.losses import (ce_dice_loss, edl_binary_loss, edl_uncertainty_from_evidence,
                          soft_cldice_loss, focal_tversky_loss,
                          topo_loss_ph_proxy, dmt_loss_proxy)
from methods.postproc import morph_postproc, violation_correct_postproc
from methods.certify import certify_mask
from methods.ours import RepairHeadV1, RepairHeadV2, run_ours_repair

from metrics import dice_iou, sens_spec, auc_aupr
from metrics.boundary import hd95, asd_assd, bf1_multi
from metrics.skeleton import cldice_metric, skeleton_dice
from metrics.topology import topo_stats, betti_matching_error
from metrics.uncertainty import ece, overlap_iou_recall_budget, error_alignment_auc
from metrics.efficiency import changed_frac

from utils.repro import set_seed
from utils.io import ensure_dir, save_json, append_row_to_csv
from utils.timer import CUDATimer


def _prob_to_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


# ---------- memory-safe inference for large test images ----------
def _tile_positions(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos

@torch.no_grad()
def forward_maybe_tiled(model, img: torch.Tensor, max_side: int = 1024, tile: int = 512, overlap: float = 0.5, pad_multiple: int = 32):
    """Forward pass with simple sliding-window tiling (eval only).

    img: (1,C,H,W)
    Returns: (1,OutC,H,W)

    Notes:
      - Many encoder/decoder backbones (e.g., UNet++, Swin-UNet) can produce
        off-by-one spatial sizes on odd-resolution inputs due to pooling/upsampling.
        We avoid this by reflect-padding the input to a multiple of `pad_multiple`
        and then cropping the output back.
    """
    _, _, H, W = img.shape

    name = getattr(model, "__class__", type("x",(object,),{})).__name__.lower()
    mod  = getattr(model, "__module__", "").lower()
    is_heavy_attn = ("segformer" in name) or ("segformer" in mod) or ("swin" in name) or ("swin" in mod)
    # Transformer-style backbones can OOM on full-resolution attention during eval.
    # Be conservative: force tiling unless the image is small enough.
    max_side_eff = int(max_side)
    tile_eff = int(tile)
    if is_heavy_attn:
        max_side_eff = min(max_side_eff, 384)
        tile_eff = min(tile_eff, 256)
    def _pad_to_multiple(x: torch.Tensor):
        mh = int(pad_multiple)
        mw = int(pad_multiple)
        pad_h = (mh - (x.shape[-2] % mh)) % mh
        pad_w = (mw - (x.shape[-1] % mw)) % mw
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    if max(H, W) <= max_side_eff:
        img_pad, pad_h, pad_w = _pad_to_multiple(img)
        with torch.cuda.amp.autocast(enabled=img_pad.is_cuda):
            out = model(img_pad)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out

    tile = tile_eff

    stride = max(1, int(tile * (1.0 - float(overlap))))
    ys = _tile_positions(H, tile, stride)
    xs = _tile_positions(W, tile, stride)

    device = img.device
    # probe output channels
    with torch.no_grad():
        probe = img[:, :, 0:min(tile, H), 0:min(tile, W)]
        ph, pw = probe.shape[-2], probe.shape[-1]
        if ph != tile or pw != tile:
            pad_h = tile - ph
            pad_w = tile - pw
            probe = torch.nn.functional.pad(probe, (0, pad_w, 0, pad_h), mode="reflect")
        probe = _pad_to_multiple(probe)[0]
        with torch.cuda.amp.autocast(enabled=probe.is_cuda):
            pred0 = model(probe)
        out_ch = int(pred0.shape[1])

    out = torch.zeros((1, out_ch, H, W), device=device)
    wgt = torch.zeros((1, 1, H, W), device=device)

    for y in ys:
        for x in xs:
            patch = img[:, :, y:y+tile, x:x+tile]
            ph, pw = patch.shape[-2], patch.shape[-1]
            if ph != tile or pw != tile:
                pad_h = tile - ph
                pad_w = tile - pw
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
            patch = _pad_to_multiple(patch)[0]
            with torch.cuda.amp.autocast(enabled=patch.is_cuda):
                pred = model(patch)
            pred = pred[:, :, :ph, :pw]
            out[:, :, y:y+ph, x:x+pw] += pred
            wgt[:, :, y:y+ph, x:x+pw] += 1.0

    out = out / torch.clamp(wgt, min=1.0)
    return out

def train_base(model, head_mode, method, dl, device, epochs=5, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for batch in tqdm(dl, desc=f"train_base ep{ep+1}/{epochs}", leave=False):
            img, mask, _ = to_device(batch, device)
            out = model(img)
            if head_mode == "standard":
                logits = out
                loss = ce_dice_loss(logits, mask)
                if method == "loss_cldice":
                    loss = loss + 0.2 * soft_cldice_loss(logits, mask)
                elif method == "loss_topoph":
                    loss = loss + 0.2 * topo_loss_ph_proxy(logits, mask)
                elif method == "loss_dmt":
                    loss = loss + 0.1 * dmt_loss_proxy(logits, mask)
            else:
                # Keep the plain EDL baseline unchanged; for ours_* we stabilize base training with an aux seg loss.
                if method == "edl":
                    loss = edl_binary_loss(out, mask)
                else:
                    p_fg, _ = edl_uncertainty_from_evidence(out)
                    logit_fg = _prob_to_logit(p_fg)
                    loss = edl_binary_loss(out, mask) + 0.5 * ce_dice_loss(logit_fg, mask, ce_weight=0.2, dice_weight=0.8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def _fg_channel(x: torch.Tensor) -> torch.Tensor:
    """Return single-channel foreground prob/score.

    - If x has shape [B, 1, H, W], return as-is.
    - If x has shape [B, C, H, W] with C>1, take the last channel as FG.
    - If x has other shape, return as-is.
    """
    if x is None:
        return x
    if x.dim() != 4:
        return x
    if x.shape[1] == 1:
        return x
    return x[:, -1:, ...]


def train_repair(base_model, head_mode, repair_head, dl, device,
                 epochs: int = 3,
                 lr: float = 1e-3,
                 tau_unc: float = 0.5,
                 thr: float = 0.5,
                 roi_topk: float = 0.15,
                 roi_dilate: int = 11,
                 w_budget: float = 0.01,
                 w_tversky: float = 0.5,
                 tversky_alpha: float = 0.7,
                 tversky_beta: float = 0.3,
                 edit_target: float = 0.02,
                 w_edit: float = 50.0):
    """
    Train repair head on an ERROR-focused ROI (uses GT during training only).

    - ROI_train = dilated(error) ∪ {uncert > tau_unc} ∪ top-k(uncert).
    - Loss combines:
        * BCE + Dice on ROI
        * focal-Tversky on ROI
        * budget term: average |refined - base| in ROI
        * edit-target term: encourages budget ≈ edit_target

    This prevents no-op collapse and handles extreme FG sparsity, while
    avoiding overly aggressive global edits.
    """
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    repair_head.train()

    opt = torch.optim.Adam(repair_head.parameters(), lr=lr)

    def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
        """Binary dilation with a kxk kernel."""
        if k <= 1:
            return x
        kernel = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype)
        pad = k // 2
        x_bin = (x > 0.5).float()
        out = torch.nn.functional.conv2d(x_bin, kernel, padding=pad)
        return (out > 0).float()

    for ep in range(int(epochs)):
        pbar = tqdm(dl, desc=f"train_repair ep{ep+1}/{epochs}", leave=False)
        for j, batch in enumerate(pbar):
            img, mask, _ = to_device(batch, device)

            # ----- get base prediction + uncertainty -----
            with torch.no_grad():
                out = base_model(img)
                if head_mode == "standard":
                    base_prob = torch.sigmoid(out)
                    uncert = torch.zeros_like(base_prob)
                else:
                    base_prob, uncert = edl_uncertainty_from_evidence(out)

                gt = (mask > 0.5).float()
                base_prob_fg = _fg_channel(base_prob)
                uncert_fg = _fg_channel(uncert)

                # binarize base for error map
                base_bin = (base_prob_fg > thr).float()
                err = (base_bin != gt).float()

                # error ROI (dilated) + uncertainty ROI
                roi_err = _dilate(err, int(roi_dilate))
                roi_unc = (uncert_fg > tau_unc).float()
                roi_train = torch.maximum(roi_err, roi_unc)

                # optional top-k uncertainty ROI
                if float(roi_topk) < 1.0:
                    B, _, H, W = uncert_fg.shape
                    k = max(1, int(float(roi_topk) * H * W))
                    flat = uncert_fg.view(B, -1)
                    thrv = torch.kthvalue(flat, flat.size(1) - k + 1, dim=1).values
                    thrv = thrv.view(B, 1, 1, 1)
                    roi_train = torch.maximum(roi_train, (uncert_fg >= thrv).float())

                roi_train = (roi_train > 0.5).float()

            # ----- forward repair inside ROI -----
            refined_prob, _, aux = run_ours_repair(
                base_model,
                head_mode,
                repair_head,
                img,
                tau_unc=tau_unc,
                thr=thr,
                iters=1,
                roi_topk=None,
                roi_override=roi_train,
            )

            eps = 1e-6

            # collapse to foreground channel
            refined_fg = _fg_channel(refined_prob)
            mask_fg = _fg_channel(mask)
            base_prob_det = aux["base_prob"]
            base_fg = _fg_channel(base_prob_det)

            # ----- losses -----
            # BCE (foreground-focused) on ROI
            bce = F.binary_cross_entropy(refined_fg, mask_fg, reduction="none")
            bce = bce * (1.0 + 4.0 * mask_fg)

            denom = roi_train.sum().clamp_min(1.0)
            bce_roi = (bce * roi_train).sum() / denom

            # Dice on ROI
            inter = (refined_fg * mask_fg * roi_train).sum()
            p_sum = (refined_fg * roi_train).sum()
            g_sum = (mask_fg * roi_train).sum()
            dice_roi = 1.0 - (2.0 * inter + eps) / (p_sum + g_sum + eps)

            # Tversky on ROI (structure-aware)
            tp = inter
            fn = ((1.0 - refined_fg) * mask_fg * roi_train).sum()
            fp = (refined_fg * (1.0 - mask_fg) * roi_train).sum()
            tversky = 1.0 - (tp + eps) / (
                tp + float(tversky_alpha) * fn + float(tversky_beta) * fp + eps
            )

            # budget = average |refined - base| within ROI
            budget = ((refined_fg - base_fg).abs() * roi_train).sum() / denom

            # edit-target penalty to avoid extreme no-op
            edit_pen = 0.0
            if edit_target is not None and edit_target > 0.0 and w_edit > 0.0:
                edit_pen = (budget - float(edit_target)) ** 2

            loss = (
                bce_roi
                + dice_roi
                + float(w_tversky) * tversky
                + float(w_budget) * budget
                + float(w_edit) * edit_pen
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(repair_head.parameters(), 5.0)
            opt.step()

            # simple changed-fraction diagnostics (thresholded)
            if j == 0:
                with torch.no_grad():
                    refined_bin = (refined_fg > thr).float()
                    base_bin_det = (base_fg > thr).float()
                    chg = (refined_bin != base_bin_det).float().mean().item()
                    pbar.set_postfix(
                        {
                            "roi_mean": float(roi_train.mean().item()),
                            "chg~": chg,
                            "loss": float(loss.item()),
                            "budget": float(budget.item()),
                        }
                    )

    return repair_head
def eval_one(model, head_mode, method, repair_head, dl, device, save_dir: Path,
             thr=0.5, tau_unc=0.5, roi_topk: float = 0.10, ours_iters=1):
    model.eval()
    if repair_head is not None: repair_head.eval()

    rows=[]
    for i, batch in enumerate(tqdm(dl, desc="eval", leave=False)):
        img, gt, meta = to_device(batch, device)
        gt_np = (gt[0,0].detach().cpu().numpy() > 0.5)

        with CUDATimer() as t1:
            out = forward_maybe_tiled(model, img)
            if head_mode=="standard":
                prob = torch.sigmoid(out)
                uncert = torch.zeros_like(prob)
            else:
                prob, uncert = edl_uncertainty_from_evidence(out)
        infer_ms = t1.ms()

        prob_np = prob[0,0].detach().cpu().numpy().clip(0,1)
        uncert_np = uncert[0,0].detach().cpu().numpy().clip(0,1)
        base_bin = prob_np > thr

        with CUDATimer() as t2:
            cert = certify_mask(base_bin)
        cert_ms = t2.ms()
        viol = cert["viol_map"].astype(np.float32)/255.0

        refined_prob_np = prob_np.copy()
        refined_bin = base_bin.copy()
        ran_rule = 0
        with CUDATimer() as t3:
            if method=="post_morph":
                refined_bin = morph_postproc(prob_np, thr=thr).astype(bool)
                refined_prob_np = refined_bin.astype(np.float32)
            elif method=="post_viol":
                refined_bin = violation_correct_postproc(prob_np, cert["viol_map"], thr=thr).astype(bool)
                refined_prob_np = refined_bin.astype(np.float32)
            elif method.startswith("ours_"):
                refined_prob, _, aux = run_ours_repair(model, head_mode, repair_head, img,
                                          tau_unc=tau_unc, roi_topk=roi_topk,
                                          thr=thr, iters=ours_iters)
                refined_prob_np = refined_prob[0,0].detach().cpu().numpy().clip(0,1)
                refined_bin = refined_prob_np > thr
                cert2 = certify_mask(refined_bin)
                # NOTE: we do NOT auto-fallback to rule correction for ours* in baseline runs.
                # (certificate is logged separately; no extra postproc here)
                
        repair_ms = t3.ms()

        dice, iou = dice_iou(refined_bin, gt_np)
        sens, spec = sens_spec(refined_bin, gt_np)
        auc, aupr = auc_aupr(refined_prob_np, gt_np)
        hd = hd95(refined_bin, gt_np)
        asd1, assd = asd_assd(refined_bin, gt_np)
        bf = bf1_multi(refined_bin, gt_np)
        cl = cldice_metric(refined_bin, gt_np)
        sd = skeleton_dice(refined_bin, gt_np)

        ts_p = topo_stats(refined_bin); ts_g = topo_stats(gt_np)
        bme = betti_matching_error(ts_p, ts_g)

        ece_v = ece(refined_prob_np, gt_np)
        err = (base_bin != gt_np)
        ua, uapr = error_alignment_auc(uncert_np, err)
        iou_u, rec_u = overlap_iou_recall_budget(uncert_np, viol, budget=0.1)

        chg = np.nan
        if method.startswith("ours_") or method.startswith("post_"):
            chg = changed_frac(refined_bin, base_bin)

        row = {
            "idx": i,
            "dice": dice, "iou": iou,
            "auc": auc, "aupr": aupr,
            "sensitivity": sens, "specificity": spec,
            "hd95": hd, "asd": asd1, "assd": assd,
            "cldice": cl, "skeleton_dice": sd,
            "n_cc": ts_p["n_cc"], "main_cc_ratio": ts_p["main_cc_ratio"],
            "holes": ts_p["holes"], "euler": ts_p["euler"], "topo_pass": ts_p["topo_pass"],
            "betti_match_err": bme,
            "ece": ece_v,
            "unc_err_auc": ua, "unc_err_aupr": uapr,
            "unc_viol_iou@0.1": iou_u, "unc_viol_recall@0.1": rec_u,
            "changed_frac": chg,
            "ran_rule": ran_rule,
            "time_infer_ms": infer_ms, "time_cert_ms": cert_ms, "time_repair_ms": repair_ms,
            **bf
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir/"metrics.csv", index=False)
    summary = df.mean(numeric_only=True).to_dict()
    summary["ran_rule_rate"] = float(df["ran_rule"].mean()) if "ran_rule" in df.columns else 0.0
    summary["n_images"] = int(len(df))
    save_json(summary, save_dir/"summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--datasets", type=str, default="drive,chase,stare")
    ap.add_argument("--backbones", type=str, default="unet")
    ap.add_argument("--methods", type=str, default="std")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--epochs_base", type=int, default=5)
    ap.add_argument("--epochs_repair", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tau_unc", type=float, default=0.5)
    ap.add_argument("--roi_topk", type=float, default=0.10)
    ap.add_argument("--roi_dilate", type=int, default=11)
    ap.add_argument("--w_budget", type=float, default=0.01)
    ap.add_argument("--w_tversky", type=float, default=0.5)
    ap.add_argument("--w_edit", type=float, default=50.0)
    ap.add_argument("--edit_target", type=float, default=0.02)
    ap.add_argument("--tversky_alpha", type=float, default=0.7)
    ap.add_argument("--tversky_beta", type=float, default=0.3)
    ap.add_argument("--ours_iters", type=int, default=1)
    ap.add_argument("--skip_if_done", action="store_true")
    args = ap.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    backbones = [s.strip() for s in args.backbones.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root = Path(args.results_root)

    for ds in datasets:
        dl_tr, dl_te = make_loaders(ds, args.data_root, args.crop, args.batch_size)
        for bb in backbones:
            for method in methods:
                for seed in seeds:
                    set_seed(seed)

                    if method in ["std", "loss_cldice", "loss_topoph", "loss_dmt", "post_morph", "post_viol"]:
                        head_mode = "standard"
                    else:
                        head_mode = "edl"

                    base_dir = results_root / "exp1" / ds / bb / method
                    seed_dir = f"seed{seed}"
                    if method.startswith("ours"):
                        seed_dir = (
                            seed_dir
                            + f"_tu{args.tau_unc}"
                            + f"_rk{args.roi_topk}"
                            + f"_rd{args.roi_dilate}"
                            + f"_wb{args.w_budget}"
                            + f"_wt{args.w_tversky}"
                            + f"_we{args.w_edit}"
                            + f"_et{args.edit_target}"
                            + f"_ta{args.tversky_alpha}"
                            + f"_tb{args.tversky_beta}"
                            + f"_it{args.ours_iters}"
                        )
                    save_dir = base_dir / seed_dir

                    ensure_dir(save_dir)
                    if args.skip_if_done and (save_dir / "summary.json").exists():
                        print(f"[skip] {save_dir} already done")
                        continue

                    # ----- train base segmentation model -----
                    model = build_backbone(bb, head_mode=head_mode).to(device)
                    model = train_base(
                        model,
                        head_mode,
                        method,
                        dl_tr,
                        device,
                        epochs=args.epochs_base,
                        lr=args.lr,
                    )
                    torch.save(model.state_dict(), save_dir / "model.pt")

                    # ----- train repair head if needed -----
                    repair_head = None
                    if method == "ours_v1":
                        repair_head = RepairHeadV1(in_ch=6).to(device)
                        repair_head = train_repair(
                            base_model=model,
                            head_mode="edl",
                            repair_head=repair_head,
                            dl=dl_tr,
                            device=device,
                            epochs=args.epochs_repair,
                            lr=args.lr,
                            tau_unc=args.tau_unc,
                            thr=args.thr,
                            roi_topk=args.roi_topk,
                            roi_dilate=args.roi_dilate,
                            w_budget=args.w_budget,
                            w_tversky=args.w_tversky,
                            tversky_alpha=args.tversky_alpha,
                            tversky_beta=args.tversky_beta,
                            edit_target=args.edit_target,
                            w_edit=args.w_edit,
                        )
                        torch.save(repair_head.state_dict(), save_dir / "repair.pt")
                    elif method == "ours_v2":
                        repair_head = RepairHeadV2(in_ch=6).to(device)
                        repair_head = train_repair(
                            base_model=model,
                            head_mode="edl",
                            repair_head=repair_head,
                            dl=dl_tr,
                            device=device,
                            epochs=args.epochs_repair,
                            lr=args.lr,
                            tau_unc=args.tau_unc,
                            thr=args.thr,
                            roi_topk=args.roi_topk,
                            roi_dilate=args.roi_dilate,
                            w_budget=args.w_budget,
                            w_tversky=args.w_tversky,
                            tversky_alpha=args.tversky_alpha,
                            tversky_beta=args.tversky_beta,
                            edit_target=args.edit_target,
                            w_edit=args.w_edit,
                        )
                        torch.save(repair_head.state_dict(), save_dir / "repair.pt")

                    # free GPU memory between evals
                    if (isinstance(device, str) and device.startswith("cuda")) or (
                        hasattr(device, "type") and device.type == "cuda"
                    ):
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                    summary = eval_one(
                        model,
                        head_mode,
                        method,
                        repair_head,
                        dl_te,
                        device,
                        save_dir,
                        thr=args.thr,
                        tau_unc=args.tau_unc,
                        roi_topk=args.roi_topk,
                        ours_iters=args.ours_iters,
                    )

                    # record ours cfg for later analysis
                    if method.startswith("ours"):
                        summary["tau_unc"] = float(args.tau_unc)
                        summary["roi_topk"] = float(args.roi_topk)
                        summary["roi_dilate"] = int(args.roi_dilate)
                        summary["w_budget"] = float(args.w_budget)
                        summary["w_tversky"] = float(args.w_tversky)
                        summary["w_edit"] = float(args.w_edit)
                        summary["edit_target"] = float(args.edit_target)
                        summary["tversky_alpha"] = float(args.tversky_alpha)
                        summary["tversky_beta"] = float(args.tversky_beta)
                        summary["ours_iters"] = int(args.ours_iters)
                    else:
                        summary["tau_unc"] = None
                        summary["roi_topk"] = None
                        summary["roi_dilate"] = None
                        summary["w_budget"] = None
                        summary["w_tversky"] = None
                        summary["w_edit"] = None
                        summary["edit_target"] = None
                        summary["tversky_alpha"] = None
                        summary["tversky_beta"] = None
                        summary["ours_iters"] = None

                    row = {"dataset": ds, "backbone": bb, "method": method, "seed": seed, **summary}
                    append_row_to_csv(row, results_root / "exp1" / "summary.csv")


if __name__ == "__main__":
    main()
