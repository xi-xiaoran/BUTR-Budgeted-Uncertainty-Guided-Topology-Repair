# analyze_miccai_sens_spec.py
# Usage example (Windows PowerShell / cmd):
#   cd D:\python\python learning\eccv\eccv2026\代码\topo_repair_pkg
#   python analyze_miccai_sens_spec.py ^
#       --data_root "D:\python\python learning\eccv\eccv2026\代码\topo_repair_pkg\data" ^
#       --models_root "D:\python\python learning\eccv\eccv2026\代码\topo_repair_pkg\models" ^
#       --results_root "D:\python\python learning\eccv\eccv2026\代码\topo_repair_pkg\results" ^
#       --datasets drive,chase,stare

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import build_backbone
from data import build_dataset
from experiments.common import collate_dsitems
from methods.losses import edl_uncertainty_from_evidence
from methods.postproc import morph_postproc, violation_correct_postproc
from methods.ours import RepairHeadV1, RepairHeadV2, run_ours_repair


# ----------------------- utils -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _tile_positions(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.no_grad()
def forward_maybe_tiled(
    model,
    img: torch.Tensor,
    max_side: int = 1024,
    tile: int = 512,
    overlap: float = 0.5,
    pad_multiple: int = 32,
):
    """
    Safe eval forward with reflect-pad to multiple-of-32 + optional tiling.
    img: (1,C,H,W) -> out: (1,OutC,H,W)
    """
    _, _, H, W = img.shape

    def _pad_to_multiple(x: torch.Tensor):
        mh = int(pad_multiple)
        mw = int(pad_multiple)
        pad_h = (mh - (x.shape[-2] % mh)) % mh
        pad_w = (mw - (x.shape[-1] % mw)) % mw
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    # small enough -> direct
    if max(H, W) <= int(max_side):
        xpad, ph, pw = _pad_to_multiple(img)
        with torch.cuda.amp.autocast(enabled=xpad.is_cuda):
            out = model(xpad)
        if ph or pw:
            out = out[:, :, :H, :W]
        return out

    # tiling
    tile = int(tile)
    stride = max(1, int(tile * (1.0 - float(overlap))))
    ys = _tile_positions(H, tile, stride)
    xs = _tile_positions(W, tile, stride)

    device = img.device

    # probe out channels
    probe = img[:, :, 0 : min(tile, H), 0 : min(tile, W)]
    ph, pw = probe.shape[-2], probe.shape[-1]
    if ph != tile or pw != tile:
        probe = F.pad(probe, (0, tile - pw, 0, tile - ph), mode="reflect")
    probe = _pad_to_multiple(probe)[0]
    with torch.cuda.amp.autocast(enabled=probe.is_cuda):
        pred0 = model(probe)
    out_ch = int(pred0.shape[1])

    out = torch.zeros((1, out_ch, H, W), device=device)
    wgt = torch.zeros((1, 1, H, W), device=device)

    for y in ys:
        for x in xs:
            patch = img[:, :, y : y + tile, x : x + tile]
            ph, pw = patch.shape[-2], patch.shape[-1]
            if ph != tile or pw != tile:
                patch = F.pad(patch, (0, tile - pw, 0, tile - ph), mode="reflect")
            patch = _pad_to_multiple(patch)[0]
            with torch.cuda.amp.autocast(enabled=patch.is_cuda):
                pred = model(patch)
            pred = pred[:, :, :ph, :pw]
            out[:, :, y : y + ph, x : x + pw] += pred
            wgt[:, :, y : y + ph, x : x + pw] += 1.0

    out = out / torch.clamp(wgt, min=1.0)
    return out


def _fg_channel(x: torch.Tensor) -> torch.Tensor:
    """Keep foreground channel as [B,1,H,W]."""
    if x.dim() == 4 and x.shape[1] > 1:
        return x[:, -1:, ...]
    return x


def _update_confusion(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = int((pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return tp, tn, fp, fn


def _sens_spec_from_counts(tp: int, tn: int, fp: int, fn: int, eps: float = 1e-6) -> Tuple[float, float]:
    sens = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)
    return float(sens), float(spec)


def _build_eval_loader(dataset_name: str, data_root: Path, split: str) -> Optional[DataLoader]:
    """
    Build split dataset, but force eval mode:
      - ds.train=False (no aug)
      - ds.crop=None (full-res)
    Return None if split not available.
    """
    try:
        ds = build_dataset(dataset_name, data_root, split, crop=None)
    except Exception:
        return None

    # Force evaluation behavior even for split=="train"
    if hasattr(ds, "train"):
        ds.train = False
    if hasattr(ds, "crop"):
        ds.crop = None

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_dsitems,
    )
    return dl


def _infer_one(
    method: str,
    head_mode: str,
    model: torch.nn.Module,
    repair_head: Optional[torch.nn.Module],
    img: torch.Tensor,
    thr: float,
    tau_unc: float,
    roi_topk: float,
    ours_iters: int,
) -> np.ndarray:
    """
    Returns binary prediction (H,W) as bool ndarray.
    """
    method_l = method.lower()

    if method_l.startswith("ours_"):
        # run our repair (uses internal certificate)
        refined_prob, _, _ = run_ours_repair(
            model,
            head_mode,
            repair_head,
            img,
            tau_unc=tau_unc,
            thr=thr,
            iters=ours_iters,
            roi_topk=roi_topk,
        )
        prob = _fg_channel(refined_prob)
        prob_np = prob[0, 0].detach().cpu().numpy()
        return (prob_np > thr)

    # base forward
    out = forward_maybe_tiled(model, img)

    if head_mode == "standard":
        prob = torch.sigmoid(out)
        prob = _fg_channel(prob)
        prob_np = prob[0, 0].detach().cpu().numpy()
        base_bin = (prob_np > thr)

        if method_l == "post_morph":
            pred_bin = morph_postproc(prob_np, thr=thr).astype(bool)
            return pred_bin
        if method_l == "post_viol":
            # violation-based correction needs viol_map; function accepts prob + viol_map
            # Internally it will compute a correction from violation mask.
            # Here it expects the "viol_map" as uint8 (0/255) or similar; we pass what it uses inside:
            #   violation_correct_postproc(prob, viol_map, thr)
            # But we don't have viol_map here; the baseline implementation computes it in exp1 via certify_mask.
            # In this codebase, violation_correct_postproc expects an explicit viol_map.
            # So we do the same as exp1: certify base_bin and use cert["viol_map"].
            from methods.certify import certify_mask  # local import to keep top-level clean

            cert = certify_mask(base_bin)
            pred_bin = violation_correct_postproc(prob_np, cert["viol_map"], thr=thr).astype(bool)
            return pred_bin

        return base_bin

    # EDL head
    prob, _unc = edl_uncertainty_from_evidence(out)
    prob = _fg_channel(prob)
    prob_np = prob[0, 0].detach().cpu().numpy()
    return (prob_np > thr)


def _load_models_for(ds: str, method: str, backbone: str, models_root: Path, device: str):
    """
    Returns (head_mode, model, repair_head)
    """
    m = method.lower()
    if m in ["std", "loss_cldice", "loss_topoph", "loss_dmt", "post_morph", "post_viol"]:
        head_mode = "standard"
    else:
        head_mode = "edl"

    model_dir = models_root / ds / m
    model_pt = model_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"Missing model file: {model_pt}")

    model = build_backbone(backbone, head_mode=head_mode).to(device)
    sd = torch.load(model_pt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    repair_head = None
    if m == "ours_v1":
        rp = model_dir / "repair.pt"
        if not rp.exists():
            raise FileNotFoundError(f"Missing repair file: {rp}")
        repair_head = RepairHeadV1().to(device)
        repair_head.load_state_dict(torch.load(rp, map_location="cpu"), strict=True)
        repair_head.eval()

    if m == "ours_v2":
        rp = model_dir / "repair.pt"
        if not rp.exists():
            raise FileNotFoundError(f"Missing repair file: {rp}")
        repair_head = RepairHeadV2().to(device)
        repair_head.load_state_dict(torch.load(rp, map_location="cpu"), strict=True)
        repair_head.eval()

    return head_mode, model, repair_head


def eval_dataset_method_all_splits(
    dataset_name: str,
    method: str,
    backbone: str,
    data_root: Path,
    models_root: Path,
    device: str,
    thr: float,
    tau_unc: float,
    roi_topk: float,
    ours_iters: int,
) -> List[Dict]:
    """
    Evaluate split-wise and also union(all) with de-dup by img_path.
    Returns a list of rows for CSV.
    """
    head_mode, model, repair_head = _load_models_for(dataset_name, method, backbone, models_root, device)

    split_order = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}
    for sp in split_order:
        dl = _build_eval_loader(dataset_name, data_root, sp)
        if dl is not None:
            loaders[sp] = dl

    rows = []

    def _eval_loader(dl: DataLoader, desc: str, dedup: Optional[Set[str]] = None):
        tp = tn = fp = fn = 0
        n_img = 0
        for batch in tqdm(dl, desc=desc, leave=False):
            img = batch.image.to(device)
            gt = batch.mask.to(device)

            # meta is list[dict] (len=1)
            meta = batch.meta[0] if isinstance(batch.meta, list) and len(batch.meta) > 0 else {}
            img_path = str(meta.get("img_path", ""))

            if dedup is not None:
                if img_path and (img_path in dedup):
                    continue
                if img_path:
                    dedup.add(img_path)

            gt_np = (gt[0, 0].detach().cpu().numpy() > 0.5)
            pred_bin = _infer_one(
                method=method,
                head_mode=head_mode,
                model=model,
                repair_head=repair_head,
                img=img,
                thr=thr,
                tau_unc=tau_unc,
                roi_topk=roi_topk,
                ours_iters=ours_iters,
            )
            a, b, c, d = _update_confusion(pred_bin, gt_np)
            tp += a
            tn += b
            fp += c
            fn += d
            n_img += 1

        sens, spec = _sens_spec_from_counts(tp, tn, fp, fn)
        return {
            "n_images": int(n_img),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": sens,
            "specificity": spec,
        }

    # per split
    for sp, dl in loaders.items():
        stats = _eval_loader(dl, desc=f"{dataset_name}/{method}/{sp}")
        row = {
            "dataset": dataset_name,
            "method": method,
            "backbone": backbone,
            "split": sp,
            "thr": thr,
            "tau_unc": tau_unc,
            "roi_topk": roi_topk,
            "ours_iters": ours_iters,
            **stats,
        }
        rows.append(row)

    # all (dedup across splits)
    seen: Set[str] = set()
    tp = tn = fp = fn = 0
    n_img = 0
    for sp in split_order:
        if sp not in loaders:
            continue
        dl = loaders[sp]
        stats_sp = _eval_loader(dl, desc=f"{dataset_name}/{method}/ALL[{sp}]", dedup=seen)
        tp += stats_sp["tp"]
        tn += stats_sp["tn"]
        fp += stats_sp["fp"]
        fn += stats_sp["fn"]
        n_img += stats_sp["n_images"]

    sens, spec = _sens_spec_from_counts(tp, tn, fp, fn)
    rows.append(
        {
            "dataset": dataset_name,
            "method": method,
            "backbone": backbone,
            "split": "all",
            "thr": thr,
            "tau_unc": tau_unc,
            "roi_topk": roi_topk,
            "ours_iters": ours_iters,
            "n_images": int(n_img),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": sens,
            "specificity": spec,
        }
    )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to data root (the folder containing DRIVE/STARE/CHASE_DB1 etc.)")
    ap.add_argument("--models_root", type=str, required=True, help="Path to models root (models/<dataset>/<method>/model.pt)")
    ap.add_argument("--results_root", type=str, default="results", help="Where to write CSV")
    ap.add_argument("--datasets", type=str, default="drive,chase,stare", help="Comma-separated: drive,chase,stare,...")
    ap.add_argument(
        "--methods",
        type=str,
        default="std,edl,loss_cldice,loss_topoph,loss_dmt,post_morph,post_viol,ours_v1,ours_v2",
        help="Comma-separated methods (must match folder names under models/<dataset>/)",
    )
    ap.add_argument("--backbone", type=str, default="unet")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tau_unc", type=float, default=0.5)
    ap.add_argument("--roi_topk", type=float, default=0.10)
    ap.add_argument("--ours_iters", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    models_root = Path(args.models_root)
    results_root = Path(args.results_root)
    ensure_dir(results_root)

    datasets = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]
    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] data_root={data_root}")
    print(f"[info] models_root={models_root}")
    print(f"[info] results_root={results_root}")

    all_rows: List[Dict] = []
    for ds in datasets:
        for m in methods:
            try:
                rows = eval_dataset_method_all_splits(
                    dataset_name=ds,
                    method=m,
                    backbone=args.backbone,
                    data_root=data_root,
                    models_root=models_root,
                    device=device,
                    thr=args.thr,
                    tau_unc=args.tau_unc,
                    roi_topk=args.roi_topk,
                    ours_iters=args.ours_iters,
                )
                all_rows.extend(rows)
            except FileNotFoundError as e:
                print(f"[skip] {ds}/{m}: {e}")
            except Exception as e:
                print(f"[error] {ds}/{m}: {repr(e)}")

    out_csv = results_root / "MICCAI_sens_spec_unet.csv"
    df = pd.DataFrame(all_rows)

    # nicer ordering
    col_order = [
        "dataset", "method", "backbone", "split",
        "n_images", "sensitivity", "specificity",
        "tp", "tn", "fp", "fn",
        "thr", "tau_unc", "roi_topk", "ours_iters",
    ]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[cols]

    df.to_csv(out_csv, index=False)
    print(f"[done] wrote: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
