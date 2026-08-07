"""
analyse_mlp_3class.py
Analysis and figure generation for the final MLP 3-class classifier.

Generates:
  figures/roc_curves_test.jpg
  figures/pr_curves_test.jpg
  figures/confusion_matrix_test.jpg
  figures/mlp_saliency.jpg

Uses the confirmed threshold boost (spontaneous probability, boost=0.03) for
all reported predictions — this reflects the actual deployed model behaviour.

Run after test_mlp_3class.py.
Usage: python analyse_mlp_3class.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling"
SPONTANEOUS_BOOST = 0.03   # confirmed via two independent CV threshold sweeps, 30 July 2026

os.makedirs(rf"{BASE}\results_v7\figures", exist_ok=True)

CLASS_NAMES = ['Baseline', 'Spontaneous', 'Stimulated']
COLOR       = '#C2185B'

# ── CONFIG ────────────────────────────────────────────────────────────────────
with open("fscv_config_ipsc.yaml") as f:
    cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * cfg['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES
V_OX_START, V_OX_END = cfg['v_oxidation_start'], cfg['v_oxidation_end']
V_RED_START, V_RED_END = cfg['v_reduction_start'], cfg['v_reduction_end']

# ── MLP ARCHITECTURE (must match train_models_3class_v5.py) ─────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)

def apply_threshold_boost(proba, boost=SPONTANEOUS_BOOST, class_idx=1):
    boosted = proba.copy()
    boosted[:, class_idx] *= boost
    return boosted / boosted.sum(axis=1, keepdims=True)

# ── LOAD TEST DATA ────────────────────────────────────────────────────────────
test_meta = pd.read_csv(rf"{BASE}\windows_metadata_test_v2.csv")
y_test    = test_meta['label'].values
print(f"Test set: {len(y_test)} samples")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name} ({i}): {(y_test == i).sum()}")

print("\nLoading raw test windows...")
X_raw = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
                  for wid in test_meta['window_id'].values], dtype=np.float32)

# ── LOAD MODEL AND GET PROBABILITIES ──────────────────────────────────────────
print("Loading MLP model (models_v7)...")
mlp_data = pickle.load(open(rf"{BASE}\models_v7\mlp_model.pkl", 'rb'))
mlp_m = MLP()
mlp_m.load_state_dict(mlp_data['model_state'])
mlp_m.eval()

X_norm = (X_raw - mlp_data['mean']) / (mlp_data['std'] + 1e-8)
with torch.no_grad():
    mlp_proba_raw = torch.softmax(mlp_m(torch.FloatTensor(X_norm)), dim=1).numpy()  # (n, 3)

mlp_proba = apply_threshold_boost(mlp_proba_raw)   # final, deployed-model predictions

y_bin = label_binarize(y_test, classes=[0, 1, 2])   # (n, 3)

# ── SUMMARY METRICS TO TERMINAL ───────────────────────────────────────────────
preds     = np.argmax(mlp_proba, axis=1)
f1        = f1_score(y_test, preds, average='macro', zero_division=0)
mcc       = matthews_corrcoef(y_test, preds)
auc_macro = roc_auc_score(y_test, mlp_proba, multi_class='ovr', average='macro')

print("\n" + "=" * 45)
print(f"MLP (boosted, FINAL)   F1_macro={f1:.4f}   MCC={mcc:.4f}   AUC_macro={auc_macro:.4f}")
print("=" * 45)

# ── 1. ROC CURVES (OvR per class) ──────────────────────────────────────────────
print("\n[1/4] ROC Curves (multiclass OvR)...")
fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], mlp_proba[:, i])
    ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_title('MLP — ROC Curves (One-vs-Rest, boosted)', fontsize=13, fontweight='bold')
ax.set_xlabel('FPR', fontsize=11)
ax.set_ylabel('TPR', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(rf"{BASE}\results_v7\figures\roc_curves_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ roc_curves_test.jpg")

# ── 2. PRECISION-RECALL CURVES (OvR per class) ────────────────────────────────
print("[2/4] Precision-Recall Curves (multiclass OvR)...")
fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
for i, cls in enumerate(CLASS_NAMES):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], mlp_proba[:, i])
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, lw=2, label=f'{cls} (AUC={pr_auc:.3f})')
    chance = y_bin[:, i].mean()
    ax.axhline(chance, linestyle=':', lw=1, alpha=0.4)
ax.set_title('MLP — Precision-Recall Curves (One-vs-Rest, boosted)', fontsize=13, fontweight='bold')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(rf"{BASE}\results_v7\figures\pr_curves_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ pr_curves_test.jpg")

# ── 3. CONFUSION MATRIX ────────────────────────────────────────────────────────
print("[3/4] Confusion Matrix...")
cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='RdPu', colorbar=False)
ax.set_title(f'MLP Confusion Matrix — Test Set (boosted, MCC={mcc:.3f})', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(rf"{BASE}\results_v7\figures\confusion_matrix_test.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ confusion_matrix_test.jpg")

# ── 4. MLP GRADIENT SALIENCY (the interpretability gate) ─────────────────────
print("[4/4] MLP Gradient Saliency...")
mlp_m.eval()
X_sal = torch.FloatTensor(X_norm)
X_sal.requires_grad_(True)

# Sum logits across all 3 classes for class-agnostic saliency
logits = mlp_m(X_sal).sum()
logits.backward()

saliency    = np.abs(X_sal.grad.detach().numpy()).mean(axis=0)
saliency_2d = saliency.reshape(N_VOLTAGE_PTS, WINDOW_FRAMES)   # (1100, n_frames)
saliency_by_voltage = saliency_2d.mean(axis=1)                 # (1100,) averaged over time

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
im = ax.imshow(saliency_2d, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(im, ax=ax, label='Mean |Gradient|')
ax.set_xlabel(f'Time Frames (×{1/cfg["fscv_hz"]:.1f}s each)', fontsize=12)
ax.set_ylabel('Voltage Points (row 0 = 0.20V, row 199 ≈ 0.60V, row 899 ≈ 0.0V)', fontsize=10)

for row, label, color in [
    (199, 'Oxidation (+0.6V)',  'cyan'),
    (899, 'Reduction (0.0V)',   'lime'),
]:
    ax.axhline(row, color=color, lw=1.5, linestyle='--', alpha=0.8)
    ax.text(WINDOW_FRAMES * 0.01, row + 8, label, color=color, fontsize=9)

ax.set_title('MLP Gradient Saliency Map — Input Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(rf"{BASE}\results_v7\figures\mlp_saliency.jpg", dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ mlp_saliency.jpg")

# Quantitative convergence check — enrichment relative to a uniform-attention baseline,
# not just raw percentage share (since the two bands don't span equal, or 50%, of the axis)
total_saliency = saliency_by_voltage.sum()
ox_frac  = saliency_by_voltage[V_OX_START:V_OX_END].sum() / total_saliency
red_frac = saliency_by_voltage[V_RED_START:V_RED_END].sum() / total_saliency

ox_width_frac  = (V_OX_END - V_OX_START) / N_VOLTAGE_PTS
red_width_frac = (V_RED_END - V_RED_START) / N_VOLTAGE_PTS
band_width_frac = ox_width_frac + red_width_frac

ox_enrichment  = ox_frac / ox_width_frac
red_enrichment = red_frac / red_width_frac
combined_enrichment = (ox_frac + red_frac) / band_width_frac

print(f"\n  Oxidation band ({V_OX_START}-{V_OX_END}): {ox_frac*100:.1f}% of saliency "
      f"vs. {ox_width_frac*100:.1f}% expected by chance  →  {ox_enrichment:.2f}x enrichment")
print(f"  Reduction band ({V_RED_START}-{V_RED_END}): {red_frac*100:.1f}% of saliency "
      f"vs. {red_width_frac*100:.1f}% expected by chance  →  {red_enrichment:.2f}x enrichment")
print(f"  Combined: {(ox_frac+red_frac)*100:.1f}% of saliency vs. {band_width_frac*100:.1f}% "
      f"expected by chance  →  {combined_enrichment:.2f}x enrichment")

print("\n" + "=" * 50)
print("INTERPRETABILITY GATE (enrichment-based, >1.0x = above-chance reliance)")
print("=" * 50)
for name, enr in [("Oxidation", ox_enrichment), ("Reduction", red_enrichment), ("Combined", combined_enrichment)]:
    status = "ABOVE CHANCE" if enr > 1.0 else "AT/BELOW CHANCE"
    print(f"  {name:10s}: {enr:.2f}x  →  {status}")

if ox_enrichment > 1.0 and red_enrichment > 1.0:
    print("\nBoth bands show above-chance saliency — full interpretability convergence.")
elif ox_enrichment > 1.0 or red_enrichment > 1.0:
    print("\nPartial convergence — only one band shows above-chance reliance.")
    print("Report which band specifically, and note this explicitly rather than")
    print("claiming full oxidation+reduction convergence.")
else:
    print("\nNeither band shows above-chance saliency — REVIEW NEEDED,")
    print("saliency does not appear to concentrate on electrochemically meaningful regions.")

print(f"\n✓ ANALYSIS COMPLETE — 4 figures saved to results_v7\\figures\\")
