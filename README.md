# iPSC 3-Class Classification Pipeline (Baseline / Spontaneous / Stimulated)

This README explains how to run the iPSC 3-class FSCV classification
pipeline end to end: from raw recordings + labels, to a trained MLP, to a
held-out test evaluation, to interpretability figures.

**MLP-only.** RF, XGBoost, and CNN were tested and dropped — RF/XGB lack
feature separability on the spontaneous/stimulated boundary (their
hand-engineered features can't access the waveform-shape information that
distinguishes the two), and CNN never outperformed the MLP. There is no
feature-engineering step in this pipeline — the MLP trains directly on raw
waveforms.

Classes: `0 = baseline`, `1 = spontaneous`, `2 = stimulated`

---

## 1. Run order

Run these scripts **in this exact order**.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `make_windows_3class.py` | raw recordings (`.npy`/`.txt`/`.csv`) + labels CSV | `window_arrays/*.npy` + `windows_metadata.csv` |
| 2 | `make_split_3class.py` | `windows_metadata.csv` | `windows_metadata_train_v2.csv`, `windows_metadata_test_v2.csv` |
| 3 | `train_models_3class_mlp.py` | `windows_metadata_train_v2.csv` + `window_arrays/` | `models_v7/mlp_model.pkl` (+ `mlp_oof_ytrue.npy`, `mlp_oof_yproba.npy`, prints CV metrics) |
| 4 | `test_mlp_3class.py` | `windows_metadata_test_v2.csv` + `window_arrays/` + `mlp_model.pkl` | `results_v7/mlp_test_unscaled.json`, `mlp_test_final.json`, `mlp_proba_raw.npy`, `mlp_proba_scaled.npy` |
| 5 | `analyse_mlp_3class.py` | `windows_metadata_test_v2.csv` + `window_arrays/` + `mlp_model.pkl` | `results_v7/figures/roc_curves_test.jpg`, `pr_curves_test.jpg`, `confusion_matrix_test.jpg`, `mlp_saliency.jpg` (+ prints interpretability gate result) |

**Important workflow rule:** steps 1–3 are where you iterate (re-run as many
times as you like while tuning). Steps 4 and 5 touch the held-out test set
and should only be run **once**, at the end, when the CV results from step 3
look good. Don't treat CV numbers and held-out test numbers as interchangeable.

`utils_3class.py` is not run directly — it's a shared helper module imported
by steps 3–5 (metrics, raw-window loading, `RANDOM_STATE`, `CLASS_NAMES`).

---

## 2. What needs changing before you run anything

Nothing here takes an input/output folder as a command-line argument — paths
are hardcoded at the top of each file.

### `make_windows_3class.py` — lines 14–16
```python
PLOT_DIR   = r"...\data for 3 class annotations"   # folder of raw recording files
LABELS_CSV = r"...\3 class output\FSCV_Labels_June.csv"   # your annotation CSV
WINDOW_DIR = r"...\3 class output after relabelling\window_arrays"   # where windowed .npy arrays get saved
```

### `make_split_3class.py`, `utils_3class.py`, `train_models_3class_mlp.py`, `test_mlp_3class.py`, `analyse_mlp_3class.py`
Each has a `BASE` constant near the top:
```python
BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling"
```
**This must be identical across all five files** — each script writes into
`BASE\...` and the next one reads from `BASE\...`. This is the single most
important path to check; if it's inconsistent, later steps won't find their
inputs.

### `fscv_config_ipsc.yaml`
Check these values match your recording setup before running step 1:
```yaml
fscv_hz: 10.0                 # sampling rate
v_oxidation_start: 200        # oxidation band row indices
v_oxidation_end: 400
v_reduction_start: 800        # reduction band row indices
v_reduction_end: 1000
balance_ratio: 2              # baseline cap, relative to larger signal class
```

---

## 3. How to run

From the folder containing all the scripts and `fscv_config_ipsc.yaml`:

```bash
python make_windows_3class.py --config fscv_config_ipsc.yaml
python make_split_3class.py --config fscv_config_ipsc.yaml
python train_models_3class_mlp.py
python test_mlp_3class.py
python analyse_mlp_3class.py
```

---

## 4. The spontaneous probability-scaling correction

The unscaled MLP systematically over-calls the spontaneous class (predicts
it far more often than its true frequency). `test_mlp_3class.py` and
`analyse_mlp_3class.py` both apply a post-hoc correction — `SPONTANEOUS_SCALE
= 0.03` — that **scales spontaneous's predicted probability down** (multiplies
it by 0.03) before renormalising and taking argmax. This was confirmed via
two independent CV threshold sweeps (30 July 2026), both converging on the
0.03–0.04 region.

This is a **calibration correction to the decision rule**, not a change to
the trained model — `mlp_model.pkl` itself is unaffected. Note this is a
downward scale, not an additive "boost": despite the constant's name history
in earlier drafts, the correction suppresses over-called spontaneous
predictions rather than amplifying them.

Both test and analysis scripts report unscaled and scaled metrics side by
side so the effect of the correction is visible, not hidden.

---

## 5. Outputs you'll end up with (inside `BASE`)

```
3 class output after relabelling/
├── window_arrays/                          (step 1)
├── windows_metadata.csv                    (step 1 — all windows)
├── windows_metadata_train_v2.csv           (step 2 — train split)
├── windows_metadata_test_v2.csv            (step 2 — held-out test split, untouched until step 4)
├── models_v7/
│   ├── mlp_model.pkl
│   ├── mlp_oof_ytrue.npy                   (out-of-fold CV predictions)
│   └── mlp_oof_yproba.npy
└── results_v7/
    ├── mlp_test_unscaled.json / mlp_test_final.json
    ├── mlp_proba_raw.npy / mlp_proba_scaled.npy
    └── figures/
        ├── roc_curves_test.jpg
        ├── pr_curves_test.jpg
        ├── confusion_matrix_test.jpg
        └── mlp_saliency.jpg
```

`mlp_test_final.json` (scaled) is the final number to report for this
bundle. `mlp_saliency.jpg` and the enrichment printout from step 5 are the
interpretability gate — report oxidation/reduction convergence honestly (as
of the last run: oxidation 1.82× above chance, reduction 0.71× at/below
chance — partial convergence only).

---

## 6. Quick sanity checks

- Step 1 print-out should show a plausible baseline/spontaneous/stimulated
  window count, all non-zero.
- Step 2 print-out shows the balanced counts and the group-aware split —
  check `Test groups` isn't empty and doesn't overlap with train groups.
- Step 3 prints CV `F1_macro` per fold and a final summary — this is your
  working number, iterate here. Target gate: macro F1 ≥ 0.80.
- Steps 4 and 5 should only be run once real CV results look acceptable,
  since the held-out test set is one-shot.
- Step 5's interpretability gate printout tells you directly whether
  convergence is full, partial, or absent — report whichever it says, don't
  round up.
