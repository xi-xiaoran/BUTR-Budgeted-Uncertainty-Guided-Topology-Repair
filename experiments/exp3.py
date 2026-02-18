from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--out", type=str, default="results/exp3/time_summary.csv")
    args = ap.parse_args()
    src = Path(args.results_root)/"exp1"/"summary.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    cols = [c for c in ["dataset","backbone","method","seed","time_infer_ms","time_cert_ms","time_repair_ms","ran_rule_rate","changed_frac","dice","cldice","topo_pass"] if c in df.columns]
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(outp, index=False)
    print(f"[SAVE] {outp}")
if __name__ == "__main__":
    main()
