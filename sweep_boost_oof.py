"""
sweep_boost_oof.py
Tunes the spontaneous/stimulated boost values against CV out-of-fold (OOF)
predictions, NOT the held-out test set.

This is the methodologically correct restructure: mlp_oof_yproba.npy /
mlp_oof_ytrue.npy were saved during train_mlp_3class.py's 5-fold CV -- each
window's OOF prediction came from a model that never saw that window during
training, but critically this data is entirely separate from the held-out
test set. Tuning boost here, then applying the chosen fixed value to the
test set exactly once, keeps the test set honest -- consistent with the
"RUN ONCE" principle test_mlp_3class.py's docstring already commits to.

Usage: python sweep_boost_oof.py
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

BASE = r"C:\Users\julie\OneDrive - Imperial College London\3 class output retrain"
CLASS_NAMES = ['Baseline', 'Spontaneous', 'Stimulated']

print("Loading CV out-of-fold predictions (training-side, not the test set)...")
y_oof = np.load(rf"{BASE}\models\mlp_oof_ytrue.npy")
proba_oof = np.load(rf"{BASE}\models\mlp_oof_yproba.npy")

assert len(y_oof) == len(proba_oof), "OOF label/probability count mismatch"
print(f"Loaded {len(y_oof)} out-of-fold windows\n")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name} ({i}): {(y_oof == i).sum()}")


def apply_2d_boost(proba, spont_boost, stim_boost):
    boosted = proba.copy()
    boosted[:, 1] *= spont_boost
    boosted[:, 2] *= stim_boost
    return boosted / boosted.sum(axis=1, keepdims=True)


# Sanity check: unboosted OOF F1 (won't exactly match the test set's 0.7129 --
# different data -- but should be broadly comparable, confirming this is the
# right file)
preds_unboosted = np.argmax(proba_oof, axis=1)
f1_unboosted = f1_score(y_oof, preds_unboosted, average='macro', zero_division=0)
print(f"\nUnboosted OOF F1_macro: {f1_unboosted:.4f} (CV estimate, not directly "
      f"comparable to the 0.7129 test-set number -- different data)\n")

# Coarse grid, wide enough to catch an edge-hugging optimum like the test-set
# sweep found
spont_candidates = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
stim_candidates  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]

results = []
for sb in spont_candidates:
    for tb in stim_candidates:
        proba = apply_2d_boost(proba_oof, sb, tb)
        preds = np.argmax(proba, axis=1)
        f1_macro = f1_score(y_oof, preds, average='macro', zero_division=0)
        f1_per_class = f1_score(y_oof, preds, average=None, zero_division=0, labels=[0, 1, 2])
        results.append((sb, tb, f1_macro, *f1_per_class))

results.sort(key=lambda r: -r[2])

print(f"{'Spont':>6} {'Stim':>6} {'F1_macro':>10} {'Baseline':>10} {'Spontaneous':>12} {'Stimulated':>11}")
print("-" * 62)
for sb, tb, f1m, f0, f1c, f2 in results[:15]:
    print(f"{sb:>6.2f} {tb:>6.2f} {f1m:>10.4f} {f0:>10.4f} {f1c:>12.4f} {f2:>11.4f}")

best = results[0]
sb_best, tb_best = best[0], best[1]
print(f"\nCV-tuned best: spont_boost={sb_best}, stim_boost={tb_best} -> OOF F1_macro={best[2]:.4f}")

# Flag if the winner is sitting at a grid boundary -- means a finer/wider
# sweep is still needed before this is trustworthy
spont_at_edge = sb_best in (min(spont_candidates), max(spont_candidates))
stim_at_edge  = tb_best in (min(stim_candidates), max(stim_candidates))
if spont_at_edge or stim_at_edge:
    print("\n*** WARNING: best value sits at a grid boundary -- extend the range")
    print("    and re-run before treating this as final. ***")
else:
    print("\nBest value is interior to the grid -- likely a genuine peak, not a boundary artefact.")

proba_best = apply_2d_boost(proba_oof, sb_best, tb_best)
preds_best = np.argmax(proba_best, axis=1)
cm = confusion_matrix(y_oof, preds_best, labels=[0, 1, 2])
print(f"\nOOF confusion matrix at spont_boost={sb_best}, stim_boost={tb_best}:")
print(f"{'':14}", "  ".join(f"{c:>14}" for c in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:14}", "  ".join(f"{v:>14}" for v in row))

print(f"\n{'='*70}")
print("NEXT STEP: set SPONTANEOUS_BOOST / STIMULATED_BOOST to the values above")
print("in test_mlp_3class.py and analyse_mlp_3class.py, then run test_mlp_3class.py")
print("ONE FINAL TIME against the held-out test set for the honest reportable number.")
print(f"{'='*70}")