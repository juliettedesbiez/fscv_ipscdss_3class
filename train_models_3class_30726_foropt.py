"""
Train RF, XGBoost, MLP, and/or CNN — 3-class classifier.
Classes: 0=baseline, 1=spontaneous, 2=stimulated

This version: MLP uses plain CrossEntropyLoss + WeightedRandomSampler
(sampler-only test — completes the 4-way loss/sampler comparison).
CNN unchanged from the clean v4 version (FocalLoss only, no sampler, no crash).

Usage: python train_models_3class_v5.py [all|rf|xgb|mlp|cnn]
Run extract_features_3class.py first.
"""

import os, sys, pickle, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils_3class import load_features, load_raw_for_features, compute_metrics, print_metrics, select_models, RANDOM_STATE

import yaml

with open("fscv_config.yaml") as f:
    _cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _cfg['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES

N_SPLITS = 5

os.makedirs(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7", exist_ok=True)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # per-class weights, same as class_weights tensor
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def main(selected=None):
    args = [a.lower() for a in sys.argv[1:]]
    if args:
        selected = ['rf', 'xgb', 'mlp', 'cnn'] if 'all' in args else [a for a in args if a in ['rf', 'xgb', 'mlp', 'cnn']]
    elif selected is None:
        selected = select_models("train")

    if not selected: print("No models selected."); return
    if not os.path.exists(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\features_v2.csv"):
        print("features_v2.csv not found — run extract_features_3class.py first"); return

    X_feat, y, groups = load_features()
    X_raw, _, _ = load_raw_for_features()

    results = {}
    if 'rf'  in selected: results['rf']  = train_rf(X_feat, y, groups)
    if 'xgb' in selected: results['xgb'] = train_xgb(X_feat, y, groups)
    if 'mlp' in selected: results['mlp'] = train_mlp(X_raw,  y, groups)
    if 'cnn' in selected: results['cnn'] = train_cnn(X_raw,  y, groups)

    print("\n" + "="*40 + "\nSUMMARY\n" + "="*40)
    for n, m in results.items():
        print(f"  {n.upper()}: F1_macro={m['f1_macro']:.4f}  AUC={m['auc_macro']:.4f}")


def train_rf(X, y, groups):
    print("\nTraining Random Forest (3-class)...")
    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     class_weight='balanced', n_jobs=-1,
                                     random_state=RANDOM_STATE)
        clf.fit(X[tr], y[tr])
        y_true_all.extend(y[te])
        y_proba_all.extend(clf.predict_proba(X[te]))   # shape (n, 3)
        print(f"  Fold {fold+1}/{N_SPLITS}")

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "RF")

    final = RandomForestClassifier(n_estimators=200, max_depth=20,
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=RANDOM_STATE)
    final.fit(X, y)
    pickle.dump({'model': final}, open(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\rf_model.pkl", 'wb'))
    return metrics


def train_xgb(X, y, groups):
    print("\nTraining XGBoost (3-class)...")
    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    # Per-class sample weights for imbalance handling
    class_counts = np.bincount(y)
    sample_weight = np.array([1.0 / class_counts[c] for c in y])
    sample_weight = sample_weight / sample_weight.mean()  # normalise

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                 objective='multi:softprob', num_class=3,
                                 random_state=RANDOM_STATE, verbosity=0)
        clf.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        y_true_all.extend(y[te])
        y_proba_all.extend(clf.predict_proba(X[te]))   # shape (n, 3)
        print(f"  Fold {fold+1}/{N_SPLITS}")

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "XGB")

    final = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                               objective='multi:softprob', num_class=3,
                               random_state=RANDOM_STATE, verbosity=0)
    final.fit(X, y, sample_weight=sample_weight)
    pickle.dump({'model': final}, open(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\xgb_model.pkl", 'wb'))
    return metrics


def train_mlp(X, y, groups):
    print(f"\nTraining MLP (3-class, input={MLP_INPUT})...")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(MLP_INPUT, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 3))          # 3 output classes
        def forward(self, x): return self.net(x)

    # Class weights for plain CrossEntropyLoss (NOT FocalLoss this run — testing sampler in isolation)
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights[1] *= 3.0   # label 1 = spontaneous, hardest class — boost weight
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr_mean = X[tr].mean(); X_tr_std = X[tr].std() + 1e-8
        X_tr = (X[tr] - X_tr_mean) / X_tr_std
        X_te = (X[te] - X_tr_mean) / X_tr_std

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y[tr])   # loss needs LongTensor
        X_te_t = torch.FloatTensor(X_te)

        # Sampler is the sole imbalance-correction mechanism this run (loss is plain/weighted CrossEntropyLoss above)
        tr_class_counts = np.bincount(y[tr])
        sample_weights = 1.0 / tr_class_counts[y[tr]]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        model = MLP()
        opt    = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, sampler=sampler)

        best_f1, patience, best_proba = -1.0, 0, None

        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_te_t)
                proba  = torch.softmax(logits, dim=1).numpy()  # (n, 3)
                preds  = np.argmax(proba, axis=1)
                f1     = f1_score(y[te], preds, average='macro', zero_division=0)

                print(f"    Fold {fold+1}/{N_SPLITS} Epoch {epoch+1:2d}/100 | F1_macro: {f1:.4f}")
                sched.step(f1)

                if f1 > best_f1:
                    best_f1, best_proba, patience = f1, proba.copy(), 0
                else:
                    patience += 1
                    if patience >= 15:
                        print(f"    Early stop at epoch {epoch+1}"); break

        y_true_all.extend(y[te])
        y_proba_all.extend(best_proba)
        print(f"  Fold {fold+1}/{N_SPLITS} best F1_macro={best_f1:.4f}")

    # Save OOF probabilities for threshold optimization (no retraining needed later)
    np.save(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\mlp_oof_ytrue.npy", np.array(y_true_all))
    np.save(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\mlp_oof_yproba.npy", np.array(y_proba_all))

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "MLP")

    # Final model trained on all data
    X_all_mean = X.mean(); X_all_std = X.std() + 1e-8
    X_all = (X - X_all_mean) / X_all_std

    all_class_counts = np.bincount(y)
    all_sample_weights = 1.0 / all_class_counts[y]
    all_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(all_sample_weights),
        num_samples=len(all_sample_weights),
        replacement=True
    )
    final  = MLP()
    opt    = torch.optim.Adam(final.parameters(), lr=1e-4, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_all),
                                      torch.LongTensor(y)), batch_size=32, sampler=all_sampler)
    for _ in range(30):
        final.train()
        for xb, yb in loader:
            opt.zero_grad(); criterion(final(xb), yb).backward(); opt.step()

    pickle.dump({'model_state': final.state_dict(),
                 'mean': X_all_mean, 'std': X_all_std}, 
                 open(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\mlp_model.pkl", 'wb'))
    return metrics


def train_cnn(X, y, groups):
    print(f"\nTraining 1D CNN (3-class, input={MLP_INPUT})...")

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(WINDOW_FRAMES, 32, kernel_size=15, padding=7), nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=9, padding=4), nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 3)
            )
        def forward(self, x):
            # x: (batch, N_VOLTAGE_PTS * WINDOW_FRAMES) flat -> (batch, WINDOW_FRAMES, N_VOLTAGE_PTS)
            x = x.view(-1, N_VOLTAGE_PTS, WINDOW_FRAMES).transpose(1, 2)
            return self.head(self.conv(x))

    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights[1] *= 4.0
    class_weights = class_weights / class_weights.sum()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    gkf = GroupKFold(N_SPLITS)
    y_true_all, y_proba_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr_mean = X[tr].mean(); X_tr_std = X[tr].std() + 1e-8
        X_tr = (X[tr] - X_tr_mean) / X_tr_std
        X_te = (X[te] - X_tr_mean) / X_tr_std

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.LongTensor(y[tr])
        X_te_t = torch.FloatTensor(X_te)

        # Sampler removed — FocalLoss is the sole imbalance-correction mechanism (clean v4 CNN, no crash)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

        model  = CNN()
        opt    = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

        best_f1, patience, best_proba = -1.0, 0, None

        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_te_t)
                proba  = torch.softmax(logits, dim=1).numpy()
                preds  = np.argmax(proba, axis=1)
                f1     = f1_score(y[te], preds, average='macro', zero_division=0)

                print(f"    Fold {fold+1}/{N_SPLITS} Epoch {epoch+1:2d}/100 | F1_macro: {f1:.4f}")
                sched.step(f1)

                if f1 > best_f1:
                    best_f1, best_proba, patience = f1, proba.copy(), 0
                else:
                    patience += 1
                    if patience >= 15:
                        print(f"    Early stop at epoch {epoch+1}"); break

        y_true_all.extend(y[te])
        y_proba_all.extend(best_proba)
        print(f"  Fold {fold+1}/{N_SPLITS} best F1_macro={best_f1:.4f}")

    metrics = compute_metrics(np.array(y_true_all), np.array(y_proba_all))
    print_metrics(metrics, "CNN")

    X_all_mean = X.mean(); X_all_std = X.std() + 1e-8
    X_all = (X - X_all_mean) / X_all_std

    final  = CNN()
    opt    = torch.optim.Adam(final.parameters(), lr=1e-4, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_all),
                                      torch.LongTensor(y)), batch_size=32, shuffle=True)
    for _ in range(30):
        final.train()
        for xb, yb in loader:
            opt.zero_grad(); criterion(final(xb), yb).backward(); opt.step()

    pickle.dump({'model_state': final.state_dict(),
                 'mean': X_all_mean, 'std': X_all_std},
                 open(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7\cnn_model.pkl", 'wb'))
    return metrics


if __name__ == "__main__":
    main()