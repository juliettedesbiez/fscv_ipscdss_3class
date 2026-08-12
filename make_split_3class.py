"""
Balance classes and create a group-aware 70:30 train/test split — iPSC
3-CLASS, MLP-only pipeline.

No engineered features are computed here (rise_time, decay_time,
ox_red_ratio, etc.) — those existed only for RF/XGB, which are dropped
from this pipeline. This script does only the two jobs the MLP path
still needs from extract_features_3class.py: class balancing and a
leak-free group-aware split. Output is a plain window_id/label/group_id
table, not a feature table.

Usage: python make_split_3class.py [--config fscv_config_ipsc.yaml]

Classes: 0=baseline, 1=spontaneous, 2=stimulated
"""

import argparse
import numpy as np
import pandas as pd
import yaml
from utils_3class import RANDOM_STATE

BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output retrain"


def load_config(path="fscv_config_ipsc.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='fscv_config_ipsc.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    balance_ratio = cfg['balance_ratio']

    print("Loading ALL windows...")
    meta = pd.read_csv(rf"{BASE}\windows_metadata.csv")

    print(f"Total: {len(meta)} windows")
    print(f"  Baseline:    {(meta['label']==0).sum()}")
    print(f"  Spontaneous: {(meta['label']==1).sum()}")
    print(f"  Stimulated:  {(meta['label']==2).sum()}")

    # BALANCE: keep all spontaneous + stimulated, cap baseline at
    # balance_ratio x the larger of the two signal classes
    signal = meta[meta['label'].isin([1, 2])]
    baseline = meta[meta['label'] == 0]

    n_cap = balance_ratio * max((signal['label']==1).sum(), (signal['label']==2).sum())
    np.random.seed(RANDOM_STATE)
    baseline = baseline.sample(min(len(baseline), n_cap), random_state=RANDOM_STATE)

    balanced = pd.concat([signal, baseline]).reset_index(drop=True)

    print(f"\nBalanced set: {len(balanced)} windows")
    print(f"  Baseline:    {(balanced['label']==0).sum()}")
    print(f"  Spontaneous: {(balanced['label']==1).sum()}")
    print(f"  Stimulated:  {(balanced['label']==2).sum()}")

    # SPLIT: 70/30 on balanced data, GROUP-AWARE (whole recordings go to
    # train or test, never split)
    print("\nSplitting 70% train / 30% test (group-aware)...")
    np.random.seed(RANDOM_STATE)
    groups = balanced['group_id'].unique()
    np.random.shuffle(groups)
    n_test_groups = max(1, int(len(groups) * 0.3))
    test_groups = set(groups[:n_test_groups])
    balanced['split'] = balanced['group_id'].apply(lambda g: 'test' if g in test_groups else 'train')

    print(f"  Train: {(balanced['split']=='train').sum()}")
    print(f"  Test: {(balanced['split']=='test').sum()}")
    print(f"  Train groups: {balanced[balanced['split']=='train']['group_id'].nunique()}")
    print(f"  Test groups ({n_test_groups}): {sorted(test_groups)}")

    train = balanced[balanced['split'] == 'train'][['window_id', 'file_id', 'group_id', 'label']]
    test  = balanced[balanced['split'] == 'test'][['window_id', 'file_id', 'group_id', 'label']]

    train.to_csv(rf"{BASE}\windows_metadata_train_v2.csv", index=False)
    test.to_csv(rf"{BASE}\windows_metadata_test_v2.csv", index=False)

    print(f"\n✓ {BASE}\\windows_metadata_train_v2.csv ({len(train)} windows)")
    print(f"✓ {BASE}\\windows_metadata_test_v2.csv ({len(test)} windows)")


if __name__ == "__main__":
    main()
