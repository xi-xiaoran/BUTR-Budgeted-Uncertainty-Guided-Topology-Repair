from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from backbone import build_backbone
from experiments.common import make_loaders, to_device
from methods.losses import edl_uncertainty_from_evidence
from methods.certify.certificate import certify_mask
from methods.ours import RepairHeadV2, run_ours_repair

from metrics import dice_iou
from metrics.boundary import hd95, asd_assd
from metrics.skeleton import cldice_metric
from metrics.topology import topo_stats, betti_matching_error

from utils.repro import set_seed
from utils.io import ensure_dir, save_json, append_row_to_csv


def _fg_channel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] > 1:
        return x[:, -1:, ...]
    return x


@torch.no_grad()
def _compute_viol_map_from_prob(prob: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Compute binary violation map (float {0,1}) from a single-image prob map.

    prob: [1,1,H,W] on GPU/CPU
    """
    p = _fg_channel(prob)
    m = (p[0, 0].detach().float().cpu().numpy() > float(thr)).astype(np.uint8)
    res = certify_mask(m)
    vm = res.get("viol_map", None)
    if vm is None:
        vm = np.zeros_like(m, dtype=np.uint8)
    vm = (vm > 0).astype(np.float32)
    t = torch.from_numpy(vm)[None, None, ...].to(device=prob.device, dtype=prob.dtype)
    return t


def _find_exp1_ckpt_dir(exp1_root: Path, dataset: str, backbone: str, method: str, seed: int):
    base = exp1_root / dataset / backbone / method
    if not base.exists():
        return None
    cands = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(f"seed{seed}")]
    cands = [p for p in cands if (p / "model.pt").exists() and (p / "repair.pt").exists()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser("exp2: ablation on ROI guidance (UNet-only)")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--datasets", type=str, default="drive,chase,stare")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--backbone", type=str, default="unet")
    ap.add_argument("--method", type=str, default="ours_v2")
    ap.add_argument("--variants", type=str, default="unc,viol")  # A1/A2 only; union is already in exp1
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--exp1_root", type=str, default="")  # if empty, use {results_root}/exp1
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tau_unc", type=float, default=0.5)
    ap.add_argument("--roi_topk", type=float, default=0.10)
    ap.add_argument("--ours_iters", type=int, default=1)
    ap.add_argument("--skip_if_done", action="store_true")
    args = ap.parse_args()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [s.strip().lower() for s in args.variants.split(",") if s.strip()]

    # Only support the two ablation variants here (union already exists in exp1).
    allowed = {"unc", "viol"}
    for v in variants:
        if v not in allowed:
            raise ValueError(f"Unsupported variant {v}. Use only: unc, viol (union is already in exp1).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_root = Path(args.results_root)
    exp1_root = Path(args.exp1_root) if args.exp1_root else (results_root / "exp1")

    out_csv = results_root / "exp2" / "summary.csv"

    for ds in datasets:
        _, dl_te = make_loaders(ds, args.data_root, crop=256, batch_size=1)

        for seed in seeds:
            set_seed(seed)

            ckpt_dir = _find_exp1_ckpt_dir(exp1_root, ds, args.backbone, args.method, seed)
            if ckpt_dir is None:
                raise FileNotFoundError(
                    f"Cannot find exp1 checkpoints for ds={ds}, backbone={args.backbone}, "
                    f"method={args.method}, seed={seed} under {exp1_root}."
                )

            # build models (edl head for ours)
            model = build_backbone(args.backbone, head_mode="edl").to(device)
            repair = RepairHeadV2(in_ch=6).to(device)

            model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
            repair.load_state_dict(torch.load(ckpt_dir / "repair.pt", map_location=device))
            model.eval()
            repair.eval()

            for variant in variants:
                save_dir = results_root / "exp2" / ds / args.backbone / "ours_v2_ablation" / f"seed{seed}" / f"variant_{variant}"
                ensure_dir(save_dir)

                if args.skip_if_done and (save_dir / "summary.json").exists():
                    continue

                mets = {
                    "dice": [],
                    "iou": [],
                    "hd95": [],
                    "assd": [],
                    "cldice": [],
                    "betti": [],
                }

                for batch in tqdm(dl_te, desc=f"exp2 {ds} seed{seed} {variant}", leave=False):
                    img, gt, _ = to_device(batch, device)
                    gt_np = (gt[0, 0].detach().cpu().numpy() > 0.5)

                    # Prepare overrides for this variant
                    if variant == "unc":
                        H, W = img.shape[-2], img.shape[-1]
                        viol0 = torch.zeros((1, 1, H, W), device=device, dtype=img.dtype)

                        refined, _, _ = run_ours_repair(
                            base_model=model,
                            head_mode="edl",
                            repair_head=repair,
                            img=img,
                            tau_unc=args.tau_unc,
                            thr=args.thr,
                            iters=args.ours_iters,
                            roi_topk=args.roi_topk,
                            roi_override=None,
                            roi_mode="unc",
                            uncert_override=None,
                            viol_override=viol0,  # bypass internal certify
                        )

                    elif variant == "viol":
                        # compute base prob once to generate viol_map
                        out = model(img)
                        base_prob, base_unc = edl_uncertainty_from_evidence(out)
                        base_prob = _fg_channel(base_prob)
                        base_unc = _fg_channel(base_unc)

                        # resize to image (safe)
                        H, W = img.shape[-2], img.shape[-1]
                        if base_prob.shape[-2:] != (H, W):
                            base_prob = F.interpolate(base_prob, size=(H, W), mode="bilinear", align_corners=False)
                            base_unc = F.interpolate(base_unc, size=(H, W), mode="bilinear", align_corners=False)

                        viol_map = _compute_viol_map_from_prob(base_prob, thr=args.thr)
                        unc0 = torch.zeros_like(base_unc)

                        refined, _, _ = run_ours_repair(
                            base_model=model,
                            head_mode="edl",
                            repair_head=repair,
                            img=img,
                            tau_unc=args.tau_unc,
                            thr=args.thr,
                            iters=args.ours_iters,
                            roi_topk=args.roi_topk,  # will be ignored for roi_mode="viol"
                            roi_override=None,
                            roi_mode="viol",
                            uncert_override=unc0,
                            viol_override=viol_map,
                        )

                    else:
                        raise RuntimeError("unreachable")

                    pred_np = (refined[0, 0].detach().cpu().numpy() > float(args.thr))

                    d, j = dice_iou(pred_np, gt_np)
                    h = hd95(pred_np, gt_np)
                    _, a = asd_assd(pred_np, gt_np)
                    c = cldice_metric(pred_np, gt_np)

                    ps = topo_stats(pred_np)
                    gs = topo_stats(gt_np)
                    bme = betti_matching_error(ps, gs)

                    mets["dice"].append(d)
                    mets["iou"].append(j)
                    mets["hd95"].append(h)
                    mets["assd"].append(a)
                    mets["cldice"].append(c)
                    mets["betti"].append(bme)

                summary = {
                    "dataset": ds,
                    "backbone": args.backbone,
                    "method": args.method,
                    "seed": int(seed),
                    "variant": variant,
                    "N": int(len(mets["dice"])),
                    "dice": _nanmean(mets["dice"]),
                    "iou": _nanmean(mets["iou"]),
                    "hd95": _nanmean(mets["hd95"]),
                    "assd": _nanmean(mets["assd"]),
                    "cldice": _nanmean(mets["cldice"]),
                    "betti_match_err": _nanmean(mets["betti"]),
                    "thr": float(args.thr),
                    "tau_unc": float(args.tau_unc),
                    "roi_topk": float(args.roi_topk),
                    "exp1_ckpt_dir": str(ckpt_dir),
                }

                save_json(summary, save_dir / "summary.json")
                append_row_to_csv(summary, out_csv)


if __name__ == "__main__":
    main()
