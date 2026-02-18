from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def ensure_dir(p: str|Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: str|Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_row_to_csv(row: dict, csv_path: str|Path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
