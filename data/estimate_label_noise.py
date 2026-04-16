"""Estimate label noise in WebMainBench using confident learning.

Uses out-of-fold GBM probabilities to:
1. Estimate the noise transition matrix (confident learning)
2. Identify likely mislabeled examples
3. Sample high-confidence errors for manual inspection
"""

import json
import os
import random
import subprocess
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")

# Load data
print("Loading data...")
df = pd.read_csv(TRAIN_PATH)
with open(FEATURES_PATH) as f:
    meta = json.load(f)
feature_cols = meta["features"]
X = df[feature_cols].values
y = df["label"].values

gbm_params = meta.get("best_config", {})
gbm_params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1})
n_pos, n_neg = y.sum(), len(y) - y.sum()
if "scale_pos_weight" not in gbm_params:
    gbm_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

# OOF probabilities
print("Computing OOF probabilities...")
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

# --- Confident Learning ---
# Estimate per-class thresholds (average predicted prob for each given label)
t_keep = oof_probs[y == 1].mean()       # avg P(keep) for labeled-keep blocks
t_discard = oof_probs[y == 0].mean()    # avg P(keep) for labeled-discard blocks

print(f"\nConfident learning thresholds:")
print(f"  Avg P(keep) for labeled KEEP:    {t_keep:.4f}")
print(f"  Avg P(keep) for labeled DISCARD: {t_discard:.4f}")

# Confident joint: count blocks in each (given, predicted) cell
# using per-class thresholds
cj = np.zeros((2, 2), dtype=int)
for i in range(len(y)):
    given = int(y[i])
    if given == 1:  # labeled KEEP
        predicted = 1 if oof_probs[i] >= t_keep else 0
    else:  # labeled DISCARD
        predicted = 1 if oof_probs[i] >= (1 - t_discard) else 0
    cj[given][predicted] += 1

print(f"\nConfident joint matrix (given_label x predicted_label):")
print(f"  {'':>20} Pred DISCARD  Pred KEEP")
print(f"  {'Given DISCARD':>20}  {cj[0][0]:>10}  {cj[0][1]:>10}")
print(f"  {'Given KEEP':>20}  {cj[1][0]:>10}  {cj[1][1]:>10}")

# Noise estimates
noisy_discard = cj[0][1]  # labeled DISCARD but model confidently says KEEP
noisy_keep = cj[1][0]     # labeled KEEP but model confidently says DISCARD
total_noisy = noisy_discard + noisy_keep
noise_rate = total_noisy / len(y)

print(f"\nEstimated label noise:")
print(f"  Mislabeled DISCARD (should be KEEP): {noisy_discard} ({noisy_discard/sum(y==0)*100:.1f}% of DISCARD labels)")
print(f"  Mislabeled KEEP (should be DISCARD): {noisy_keep} ({noisy_keep/sum(y==1)*100:.1f}% of KEEP labels)")
print(f"  Total estimated noisy labels: {total_noisy} ({noise_rate*100:.2f}%)")

# --- Identify the most likely mislabeled examples ---
# Score = |P(keep) - label| weighted by confidence
noise_scores = np.abs(oof_probs - y)

# High confidence errors: model very sure, disagrees with label
# Labeled DISCARD but P(keep) > 0.9
false_discards = np.where((y == 0) & (oof_probs > 0.9))[0]
# Labeled KEEP but P(keep) < 0.1
false_keeps = np.where((y == 1) & (oof_probs < 0.1))[0]

print(f"\nExtreme disagreements:")
print(f"  Labeled DISCARD but P(keep) > 0.9: {len(false_discards)} blocks")
print(f"  Labeled KEEP but P(keep) < 0.1:    {len(false_keeps)} blocks")

# --- Sample for manual inspection ---
print("\n" + "=" * 80)
print("SAMPLE: Labeled DISCARD but model says KEEP (P > 0.95)")
print("=" * 80)

very_false_discards = np.where((y == 0) & (oof_probs > 0.95))[0]
random.seed(42)
sample = random.sample(list(very_false_discards), min(15, len(very_false_discards)))

for idx in sample:
    row = df.iloc[idx]
    prob = oof_probs[idx]
    text_len = int(row["text_len"])
    link_ratio = row["link_ratio"]
    tag_type = int(row["tag_type"])
    position = row["position"]
    tag_names = {0: "p", 1: "heading", 2: "li", 3: "pre", 4: "td", 5: "blockquote", 6: "other"}
    print(f"\n  idx={idx} P(keep)={prob:.3f} tag={tag_names.get(tag_type,'?')} "
          f"text_len={text_len} link_ratio={link_ratio:.2f} pos={position:.2f}")

print("\n" + "=" * 80)
print("SAMPLE: Labeled KEEP but model says DISCARD (P < 0.05)")
print("=" * 80)

very_false_keeps = np.where((y == 1) & (oof_probs < 0.05))[0]
sample2 = random.sample(list(very_false_keeps), min(15, len(very_false_keeps)))

for idx in sample2:
    row = df.iloc[idx]
    prob = oof_probs[idx]
    text_len = int(row["text_len"])
    link_ratio = row["link_ratio"]
    tag_type = int(row["tag_type"])
    position = row["position"]
    tag_names = {0: "p", 1: "heading", 2: "li", 3: "pre", 4: "td", 5: "blockquote", 6: "other"}
    print(f"\n  idx={idx} P(keep)={prob:.3f} tag={tag_names.get(tag_type,'?')} "
          f"text_len={text_len} link_ratio={link_ratio:.2f} pos={position:.2f}")

# --- What percentage of pages have at least one noisy label? ---
print("\n" + "=" * 80)
print("PAGE-LEVEL NOISE ANALYSIS")
print("=" * 80)

page_ids = np.zeros(len(df), dtype=int)
positions = df["position"].values
pid = 0
for i in range(1, len(df)):
    if positions[i] < positions[i-1] - 0.3:
        pid += 1
    page_ids[i] = pid

extreme_noise = (y == 0) & (oof_probs > 0.9) | (y == 1) & (oof_probs < 0.1)
pages_with_noise = len(set(page_ids[extreme_noise]))
total_pages = pid + 1
print(f"  Pages with at least one extreme disagreement: {pages_with_noise}/{total_pages} ({pages_with_noise/total_pages*100:.1f}%)")

# Distribution of noisy blocks per page
from collections import Counter
noise_per_page = Counter(page_ids[extreme_noise])
counts = list(noise_per_page.values())
if counts:
    print(f"  Noisy blocks per affected page: mean={np.mean(counts):.1f}, median={np.median(counts):.0f}, max={max(counts)}")
