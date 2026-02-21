# BUTR: Budgeted Uncertainty-guided Topology Repair

**BUTR** is a post-hoc refinement framework for thin-structure medical segmentation (e.g., retinal vessels).  
It performs **ROI-bounded** edits guided by **pixel-wise uncertainty** and a **structure/topology violation map**, so the final prediction is **identical to the backbone outside the ROI**.

<p align="center">
  <img src="assets/method.png" width="920" alt="BUTR overview / pipeline"/>
</p>

<p align="center">
  <img src="assets/qualitative.png" width="920" alt="Qualitative comparisons"/>
</p>

---

## What’s in this repo

- **Backbones**: U-Net (primary), plus additional backbones kept for extensibility.
- **Methods**:
  - `Std`: standard segmentation training (e.g., BCE+Dice)
  - `EDL`: evidential head for uncertainty estimation
  - `Loss+clDice`, `Loss+TopoPH`, `Loss+DMT`, `Loss+clce`: training-time topology objectives
  - `Post-Morph`, `Post-Viol`: post-processing baselines
  - `BUTR-V1/V2`: ROI-constrained repair (V2 uses an additional gate)
- **Experiments**:
  - `experiments/exp1.py`: multi-method comparison
  - `experiments/exp2.py`: ablation (ROI guidance variants)
  - `experiments/exp3.py`: helper/aggregation utilities

> Note: datasets and trained weights are **not** included in this repo.

---

## Quick start

### 1) Environment
```bash
pip install -r requirements.txt
```

### 2) Prepare datasets
Place datasets under `data/` (or set your dataset root path according to the dataset loader you use).  
Check `data/` and `experiments/common.py` for expected folder structure / path settings.

### 3) Run experiments
All experiment scripts support `--help` for exact arguments:
```bash
python experiments/exp1.py --help
python experiments/exp2.py --help
python analyze_miccai_sens_spec.py --help
```

---

## ROI violation detector toggle (for ablations)

The violation/structure detector can be toggled via an environment variable:

- **Default (enabled):** `TOPO_CERTIFY=1` (or unset)
- **Disable (for ablation):**
  - Linux/macOS:
    ```bash
    export TOPO_CERTIFY=0
    ```
  - PowerShell:
    ```powershell
    $env:TOPO_CERTIFY="0"
    ```

---

## Repository layout

```
.
├── assets/                 # README figures (method/qualitative)
├── backbone/               # segmentation backbones
├── data/                   # dataset loaders (expects DRIVE / CHASE_DB1 / STARE)
├── experiments/            # exp1 / exp2 / exp3
├── methods/                # BUTR + baselines
│   ├── certify/            # structure/topology certificate & violation map
│   ├── losses/             # dice, cldice, topoph proxies, dmt proxy, edl
│   └── postproc/           # morphology / violation-correction postproc
├── metrics/                # pixel/boundary/skeleton/topology/uncertainty metrics
├── utils/                  # io, timers, reproducibility helpers
└── requirements.txt
```

---

## License

See `LICENSE`.

---

## Contact

If you run into issues reproducing results, open a GitHub Issue with:
- your OS + Python/PyTorch versions
- the exact command you ran
- the full error log
