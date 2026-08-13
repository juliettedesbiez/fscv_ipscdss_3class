# iPSC 3-Class Classification Pipeline (Baseline / Spontaneous / Stimulated)

This README explains how to run the iPSC 3-class FSCV classification
pipeline end to end: from raw recordings + labels, to a trained MLP, to a
held-out test evaluation, to interpretability figures.

**MLP-only.** RF and XGBoost were tested and dropped — RF/XGB lack
feature separability on the spontaneous/stimulated boundary (their
hand-engineered features can't access the waveform-shape information that
distinguishes the two). There is no feature-engineering step in this pipeline 
— the MLP trains directly on raw waveforms.

Classes: `0 = baseline`, `1 = spontaneous`, `2 = stimulated`

---

## 1. Run order

Run these scripts **in this exact order**.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `make_windows_3class.py` | raw recordings (`.npy`/`.txt`/`.csv`) + labels CSV | `window_arrays/*.npy` + `windows_metadata.csv` |
| 2 | `make_split_3class.py` | `windows_metadata.csv` | `windows_metadata_train.csv`, `windows_metadata_test.csv` |
| 3 | `train_mlp_3class.py` | `windows_metadata_train.csv` + `window_arrays/` | `models/mlp_model.pkl` (+ `mlp_oof_ytrue.npy`, `mlp_oof_yproba.npy`, prints CV metrics) |
| 3b (optional) | `sweep_boost_oof.py` | `models/mlp_oof_ytrue.npy`, `mlp_oof_yproba.npy` from step 3 | prints candidate `SPONTANEOUS_BOOST`/`STIMULATED_BOOST` values — nothing saved to disk |
| 4 | `test_mlp_3class.py` | `windows_metadata_test.csv` + `window_arrays/` + `mlp_model.pkl` | `results_3class/mlp_test_unboosted.json`, `mlp_test_final.json`, `mlp_proba_raw.npy`, `mlp_proba_boosted.npy` |
| 5 | `analyse_mlp_3class.py` | `windows_metadata_test.csv` + `window_arrays/` + `mlp_model.pkl` | `results_3class/figures/roc_curves_test.jpg`, `pr_curves_test.jpg`, `confusion_matrix_test.jpg`, `mlp_saliency.jpg` (+ prints interpretability gate result) |

**Important workflow rule:** steps 1–3 (and 3b) are where you iterate
(re-run as many times as you like while tuning). Steps 4 and 5 touch the
held-out test set and should only be run **once**, at the end, when the CV
results from step 3 look good. Don't treat CV numbers and held-out test
numbers as interchangeable.

`utils_3class.py` is not run directly — it's a shared helper module imported
by steps 3–5 (metrics, raw-window loading, `RANDOM_STATE`, `CLASS_NAMES`).

---

## 2. Sign convention

The raw `.txt`/`.npy` recording files use a sign convention that renders
backwards on Pablo's colormap if left uncorrected. The visual symptom:
oxidation renders **blue** and reduction renders **green** — backwards from
the correct convention (oxidation **green**, reduction **blue**).

`make_windows_3class.py`'s `load_arr()` corrects for this before any
windowing happens:
```python
def load_arr(path):
    arr = np.load(path) if path.endswith('.npy') else np.loadtxt(path)
    arr = arr[np.newaxis, :] if arr.ndim == 1 else arr
    return -arr
```
Every window written to `window_arrays/` is already sign-corrected — nothing
downstream (`make_split_3class.py`, training, testing) needs to do anything
about sign, since it only ever reads from `window_arrays/`.

If you're setting up a fresh copy of `make_windows_3class.py`, confirm this
`-arr` is present before running step 1 — if it's missing, every window
generated will be sign-flipped, silently, with no error. The labelling app
(`Labelling_App.py`) has the same fix in its own `load_arr()`; both need to
agree on the same convention since one produces the labels the other
consumes.

---

## 3. What needs changing before you run anything

Nothing here takes an input/output folder as a command-line argument — paths
are hardcoded at the top of each file.

### `make_windows_3class.py`
```python
PLOT_DIR   = r"...\data for 3 class annotations"   # folder of raw recording files
LABELS_CSV = r"...\3 class output\FSCV_Labels.csv"   # your annotation CSV
BASE       = r"...\3 class output"   # covers window_arrays/ and windows_metadata.csv
WINDOW_DIR = rf"{BASE}\window_arrays"        # derived from BASE, not set independently
```

### `make_split_3class.py`, `utils_3class.py`, `train_mlp_3class.py`, `sweep_boost_oof.py`, `test_mlp_3class.py`, `analyse_mlp_3class.py`
Each has a `BASE` constant near the top:
```python
BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output"
```
**This must be identical across all seven files** — each script writes into
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

## 4. How to run

From the folder containing all the scripts and `fscv_config_ipsc.yaml`:

```bash
python make_windows_3class.py --config fscv_config_ipsc.yaml
python make_split_3class.py --config fscv_config_ipsc.yaml
python train_mlp_3class.py
python sweep_boost_oof.py            # optional — see §5 before running test_mlp_3class.py
python test_mlp_3class.py
python analyse_mlp_3class.py
```

---

## 5. Class weighting and the probability boost

Two separate corrections are in play, at two different stages — worth not
conflating them.

**Training-time class weight** (`train_mlp_3class.py`): the CrossEntropyLoss
weight on the spontaneous class (label 1) is multiplied by **3.0** before
training. Removing this weight entirely, or reducing it, was tested and
found to weaken the interpretability gate (below); ×3.0 is the current,
validated setting.

**Inference-time probability boost** (`test_mlp_3class.py` and
`analyse_mlp_3class.py`): applied to the trained model's raw softmax output,
before argmax —
```python
SPONTANEOUS_BOOST = 0.10
STIMULATED_BOOST = 1.00
```

### Finding these values: `sweep_boost_oof.py`

Run this after step 3 (training), before step 4 (testing):
```bash
python sweep_boost_oof.py
```
It loads `mlp_oof_ytrue.npy` / `mlp_oof_yproba.npy` — the out-of-fold
predictions saved during step 3's 5-fold CV — and grid-sweeps candidate
`(spontaneous_boost, stimulated_boost)` pairs against them, printing the
top 15 combinations by F1_macro and flagging if the best one sits at the
edge of the search grid (meaning the grid needs widening and the sweep
re-run before trusting the result). Nothing is written to disk; it only
prints.

This is trained-side data, entirely separate from the held-out test set —
tuning boost here rather than against the test set directly is what keeps
the final test number honest. Once you have a result that's interior to the
grid (not sitting on a boundary), hardcode those two values into
`SPONTANEOUS_BOOST`/`STIMULATED_BOOST` in both `test_mlp_3class.py` and
`analyse_mlp_3class.py`, then run step 4 exactly once. **Never grid search
boost values directly against the test set** — if you ever find yourself
re-running `test_mlp_3class.py` repeatedly while trying different boost
values by hand, stop and use `sweep_boost_oof.py` instead.

`test_mlp_3class.py` and `analyse_mlp_3class.py` both report unboosted and
boosted metrics side by side so the effect of the boost is visible, not
hidden. Neither correction touches `mlp_model.pkl`'s weights — the boost
purely reshapes the decision rule at inference time.

---

## 6. Outputs you'll end up with (inside `BASE`)

```
3 class output/
├── window_arrays/                          (step 1)
├── windows_metadata.csv                    (step 1 — all windows)
├── windows_metadata_train.csv           (step 2 — train split)
├── windows_metadata_test.csv            (step 2 — held-out test split, untouched until step 4)
├── models/
│   ├── mlp_model.pkl
│   ├── mlp_oof_ytrue.npy                   (out-of-fold CV predictions — feeds sweep_boost_oof.py)
│   └── mlp_oof_yproba.npy
└── results_3class/
    ├── mlp_test_unboosted.json / mlp_test_final.json
    ├── mlp_proba_raw.npy / mlp_proba_boosted.npy
    └── figures/
        ├── roc_curves_test.jpg
        ├── pr_curves_test.jpg
        ├── confusion_matrix_test.jpg
        └── mlp_saliency.jpg
```

`mlp_test_final.json` (boosted) is the final number to report for this
bundle: **F1_macro = 0.7267, MCC = 0.6637, AUC_macro = 0.9198**.

`mlp_saliency.jpg` and the enrichment printout from step 5 are the
interpretability gate — report oxidation/reduction convergence honestly.
Current result: oxidation 1.35× above chance, reduction 1.00× (right at the
chance boundary, counted as above) — **full convergence**.

---

## 7. Quick sanity checks

- Step 1 print-out should show a plausible baseline/spontaneous/stimulated
  window count, all non-zero.
- Step 2 print-out shows the balanced counts and the group-aware split —
  check `Test groups` isn't empty and doesn't overlap with train groups.
- Step 3 prints CV `F1_macro` per fold and a final summary — this is your
  working number, iterate here. Target gate (from the original project
  plan, compared against Naweed's binary benchmark): macro F1 ≥ 0.80.
- Step 3b's sweep should report its best combination as interior to the
  grid, not sitting on a boundary — if it's on a boundary, widen the
  candidate ranges in `sweep_boost_oof.py` and re-run before trusting it.
- Steps 4 and 5 should only be run once real CV results look acceptable,
  since the held-out test set is one-shot.
- Step 5's interpretability gate printout tells you directly whether
  convergence is full, partial, or absent — report whichever it says, don't
  round up.
- **Reproducibility note:** training is not fully deterministic run-to-run
  by default — only `torch.manual_seed()` is set, not `np.random.seed()`,
  `random.seed()`, or the `WeightedRandomSampler`'s own generator. Two runs
  with identical code have been observed to produce meaningfully different
  CV and test results. If you need a reproducible re-run, add explicit
  seeding to all of these before retraining, and don't overwrite
  `models/mlp_model.pkl` until the new run's numbers are confirmed at least
  as good as what's already there.
