"""Analyze where GBM errors concentrate by confidence bucket."""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")

df = pd.read_csv(TRAIN_PATH)
with open(FEATURES_PATH) as f:
    meta = json.load(f)
feature_cols = meta["features"]
X = df[feature_cols].values
y = df["label"].values

n_pos = y.sum()
n_neg = len(y) - n_pos
gbm_params = meta.get("best_config", {})
gbm_params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1})
if "scale_pos_weight" not in gbm_params:
    gbm_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

# OOF probs
oof_probs = np.zeros(len(df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_cols)
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_cols, reference=dtrain)
    model = lgb.train(
        gbm_params, dtrain, num_boost_round=3083,
        valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)],
    )
    oof_probs[val_idx] = model.predict(X[val_idx])
    print(f"Fold {fold+1}: {model.best_iteration} rounds", flush=True)

preds = (oof_probs > 0.5).astype(int)
errors = preds != y
confidence = np.abs(oof_probs - 0.5) * 2  # 0 = borderline, 1 = certain

# Bucket analysis
buckets = [
    ("very low (0.0-0.2)", 0.0, 0.2),
    ("low (0.2-0.4)", 0.2, 0.4),
    ("medium (0.4-0.6)", 0.4, 0.6),
    ("high (0.6-0.8)", 0.6, 0.8),
    ("very high (0.8-1.0)", 0.8, 1.0),
]

print(f"\n{'Confidence bucket':<25} {'Blocks':>8} {'Errors':>8} {'Error%':>8} {'% of all errors':>16}")
print("-" * 70)
total_errors = errors.sum()
for name, lo, hi in buckets:
    mask = (confidence >= lo) & (confidence < hi)
    n = mask.sum()
    e = errors[mask].sum()
    pct = e / max(n, 1) * 100
    pct_of_all = e / max(total_errors, 1) * 100
    print(f"{name:<25} {n:>8} {e:>8} {pct:>7.1f}% {pct_of_all:>15.1f}%")

print(f"\nTotal: {len(df)} blocks, {total_errors} errors ({total_errors/len(df)*100:.2f}%)")

# Also look at raw probability buckets
print(f"\n{'P(keep) bucket':<25} {'Blocks':>8} {'True KEEP':>10} {'True DISC':>10} {'Error%':>8}")
print("-" * 65)
prob_buckets = [
    ("0.0-0.1", 0.0, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.3", 0.2, 0.3),
    ("0.3-0.4", 0.3, 0.4),
    ("0.4-0.5", 0.4, 0.5),
    ("0.5-0.6", 0.5, 0.6),
    ("0.6-0.7", 0.6, 0.7),
    ("0.7-0.8", 0.7, 0.8),
    ("0.8-0.9", 0.8, 0.9),
    ("0.9-1.0", 0.9, 1.001),
]
for name, lo, hi in prob_buckets:
    mask = (oof_probs >= lo) & (oof_probs < hi)
    n = mask.sum()
    keep = y[mask].sum()
    disc = n - keep
    e = errors[mask].sum()
    pct = e / max(n, 1) * 100
    print(f"{name:<25} {n:>8} {keep:>10} {disc:>10} {pct:>7.1f}%")

# High-confidence errors specifically
print(f"\nHigh-confidence errors (confidence > 0.8):")
hc_mask = confidence > 0.8
hc_errors = errors & hc_mask
n_hc_errors = hc_errors.sum()
print(f"  {n_hc_errors} blocks ({n_hc_errors/total_errors*100:.1f}% of all errors)")

# Break down: false keeps vs false discards in high-confidence
hc_false_keep = (hc_errors & (preds == 1)).sum()
hc_false_disc = (hc_errors & (preds == 0)).sum()
print(f"  False KEEP (confident boilerplate kept):  {hc_false_keep}")
print(f"  False DISCARD (confident content dropped): {hc_false_disc}")
