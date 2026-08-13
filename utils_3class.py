"""Shared utilities for FSCV classification pipeline — iPSC 3-CLASS, MLP-only.

It reads the lightweight
balance+split file produced by make_split_3class.py instead.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
CLASS_NAMES  = ['Baseline', 'Spontaneous', 'Stimulated']

BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output"


def load_raw_for_features(split="train"):
    """
    Load raw flattened windows for the given split ('train' or 'test').
    Defaults to 'train' — existing calls with no argument keep working
    unchanged.

    Reads window_id/label/group_id from the lightweight split file
    (windows_metadata_train.csv / windows_metadata_test.csv,
    produced by make_split_3class.py) rather than features.csv —
    no engineered features are computed or needed for the MLP.
    """
    print(f"Loading raw windows ({split})...")
    df = pd.read_csv(rf"{BASE}\windows_metadata_{split}.csv")
    X = np.array([np.load(rf"{BASE}\window_arrays\{wid}.npy").flatten()
              for wid in df['window_id']], dtype=np.float32)
    y, groups = df['label'].values, df['group_id'].astype(str).values
    print(f"  {len(y)} windows, {X.shape[1]} raw features, {len(np.unique(groups))} groups")
    return X, y, groups


def compute_metrics(y_true, y_proba):
    """
    Compute 3-class metrics from probability array (n_samples x 3).
    Returns per-class F1, macro F1, weighted F1, macro AUC, confusion matrix.
    """
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_per_class': {
            CLASS_NAMES[i]: float(f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0))
            for i in range(3)
        },
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro':    float(recall_score(y_true, y_pred, average='macro',    zero_division=0)),
        'auc_macro': float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    }
    return metrics


def print_metrics(metrics, name):
    """Print multiclass metrics to terminal."""
    print(f"\n{name} RESULTS")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  AUC Macro:   {metrics['auc_macro']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print("  Per-class F1:")
    for cls, f1 in metrics['f1_per_class'].items():
        print(f"    {cls}: {f1:.4f}")
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(f"    {'':14}", "  ".join(f"{c:>14}" for c in CLASS_NAMES))
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"    {CLASS_NAMES[i]:14}", "  ".join(f"{v:>14}" for v in row))
