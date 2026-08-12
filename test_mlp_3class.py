"""
Test the final MLP model on the held-out test set (RUN ONCE — final reportable numbers).
Classes: 0=baseline, 1=spontaneous, 2=stimulated
Applies the confirmed threshold boosts (spontaneous and stimulated probabilities)
before argmax, tuned via sweep_boost_3class_2d.py.

Usage: python test_mlp_3class.py
"""

import os, json, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
from utils_3class import compute_metrics, print_metrics

import yaml
with open("fscv_config_ipsc.yaml") as f:
    _cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _cfg['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES

BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output retrain"
SPONTANEOUS_BOOST = 0.10
STIMULATED_BOOST = 1.00   
RESULTS_DIR = rf"{BASE}\resultscheckofmodels"

os.makedirs(RESULTS_DIR, exist_ok=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)


def apply_threshold_boost(proba, spont_boost=SPONTANEOUS_BOOST, stim_boost=STIMULATED_BOOST,
                            spont_idx=1, stim_idx=2):
    """Scale spontaneous and stimulated probabilities and renormalise before argmax."""
    boosted = proba.copy()
    boosted[:, spont_idx] *= spont_boost
    boosted[:, stim_idx] *= stim_boost
    boosted = boosted / boosted.sum(axis=1, keepdims=True)
    return boosted


def test_mlp(X, y):
    path = rf"{BASE}\models\mlp_model.pkl"
    if not os.path.exists(path):
        print("MLP model not found"); return None, None

    data  = pickle.load(open(path, 'rb'))
    model = MLP()
    model.load_state_dict(data['model_state']); model.eval()

    X_norm = (X - data['mean']) / (data['std'] + 1e-8)
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_norm))
        proba_raw = torch.softmax(logits, dim=1).numpy()   # (n, 3), unboosted

    proba_boosted = apply_threshold_boost(proba_raw)

    print("\n=== UNBOOSTED (raw MLP output, for reference) ===")
    metrics_raw = compute_metrics(y, proba_raw)
    print_metrics(metrics_raw, 'MLP (unboosted)')

    print(f"\n=== BOOSTED (final reportable result, spont={SPONTANEOUS_BOOST}, stim={STIMULATED_BOOST}) ===")
    metrics_final = compute_metrics(y, proba_boosted)
    print_metrics(metrics_final, 'MLP (boosted, FINAL)')

    # Defensive re-creation right before writing -- guarantees the folder
    # exists at the moment it's actually needed, regardless of anything
    # that happened (or didn't) at import time.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json.dump(metrics_raw,   open(rf"{RESULTS_DIR}\mlp_test_unboosted.json", 'w'), default=float)
    json.dump(metrics_final, open(rf"{RESULTS_DIR}\mlp_test_final.json", 'w'), default=float)
    np.save(rf"{RESULTS_DIR}\mlp_proba_raw.npy", proba_raw)
    np.save(rf"{RESULTS_DIR}\mlp_proba_boosted.npy", proba_boosted)

    return metrics_raw, metrics_final


def main():
    print("\nLoading TEST set...")
    test_meta = pd.read_csv(rf"{BASE}\windows_metadata_test_v2.csv")
    y_test    = test_meta['label'].values

    print(f"Test set: {len(y_test)} samples")
    print(f"  Baseline (0):    {(y_test==0).sum()}")
    print(f"  Spontaneous (1): {(y_test==1).sum()}")
    print(f"  Stimulated (2):  {(y_test==2).sum()}")

    print("\nLoading raw test windows...")
    X_raw = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
                      for wid in test_meta['window_id'].values], dtype=np.float32)
    print(f"Raw: {X_raw.shape}\n")

    metrics_raw, metrics_final = test_mlp(X_raw, y_test)

    print("\n" + "="*40)
    if metrics_final:
        print(f"MLP FINAL: F1_macro={metrics_final['f1_macro']:.4f}  AUC={metrics_final['auc_macro']:.4f}")
    print("="*40)
    print(f"\n✓ Final result saved: {RESULTS_DIR}\\mlp_test_final.json")


if __name__ == "__main__":
    main()
