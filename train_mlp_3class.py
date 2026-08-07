"""
Train MLP — 3-class classifier.
Classes: 0=baseline, 1=spontaneous, 2=stimulated
 
RF, XGBoost, and CNN dropped from this script — RF/XGB lack feature
separability on the spontaneous/stimulated boundary (12-feature tree
models can't access the shape information the raw waveform carries),
and CNN did not outperform the MLP. MLP-only, trained on raw waveforms,
is the winning single-model choice for the 3-class bundle.
 
MLP config: plain CrossEntropyLoss + WeightedRandomSampler stacked,
lr=1e-4, ReduceLROnPlateau, 100 epochs, patience=15, class weight x3 on
spontaneous (label 1, hardest class).
 
Usage: python train_models_3class_mlp.py
Run make_split_3class.py first.
"""
 
import os, pickle, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils_3class import load_raw_for_features, compute_metrics, print_metrics, RANDOM_STATE
 
import yaml
 
with open("fscv_config_ipsc.yaml") as f:
    _cfg = yaml.safe_load(f)
WINDOW_FRAMES = int(2.0 * _cfg['fscv_hz'])
N_VOLTAGE_PTS = 1100
MLP_INPUT     = N_VOLTAGE_PTS * WINDOW_FRAMES
 
N_SPLITS = 5
 
os.makedirs(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\models_v7", exist_ok=True)
 
 
def main():
    if not os.path.exists(r"C:\Users\julie\OneDrive - Imperial College London\3 class output after relabelling\windows_metadata_train_v2.csv"):
        print("windows_metadata_train_v2.csv not found — run make_split_3class.py first"); return
 
    X_raw, y, groups = load_raw_for_features()
 
    metrics = train_mlp(X_raw, y, groups)
 
    print("\n" + "="*40 + "\nSUMMARY\n" + "="*40)
    print(f"  MLP: F1_macro={metrics['f1_macro']:.4f}  AUC={metrics['auc_macro']:.4f}")
 
 
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
 
    # Class weights for CrossEntropyLoss
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
 
        # Sampler stacked with weighted loss
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
 
 
if __name__ == "__main__":
    main()
