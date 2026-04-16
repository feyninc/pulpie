"""CRF prototype v2: train CRF properly, evaluate on actual pages.

1. Train GBM on all data (use the real model)
2. Use StratifiedKFold for OOF probs (well-calibrated)
3. Train CRF on OOF sequences
4. Apply to eval pages and compare outputs
"""

import json
import os
import subprocess
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")
HBIRD_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")

import random
random.seed(42)


def infer_page_boundaries(positions):
    page_ids = np.zeros(len(positions), dtype=int)
    pid = 0
    for i in range(1, len(positions)):
        if positions[i] < positions[i - 1] - 0.3:
            pid += 1
        page_ids[i] = pid
    return page_ids


def block_features_for_crf(row_dict, gbm_prob):
    """CRF features for one block."""
    # Quantize GBM prob
    if gbm_prob > 0.8: gq = "vhigh"
    elif gbm_prob > 0.6: gq = "high"
    elif gbm_prob > 0.4: gq = "mid"
    elif gbm_prob > 0.2: gq = "low"
    else: gq = "vlow"

    pos = row_dict["position"]
    if pos < 0.1: pq = "start"
    elif pos > 0.9: pq = "end"
    else: pq = "mid"

    lr = row_dict["link_ratio"]
    if lr > 0.5: lq = "high"
    elif lr > 0.1: lq = "mid"
    else: lq = "low"

    return {
        "gbm": gbm_prob,
        "gq": gq,
        "pos": pos,
        "pq": pq,
        "tt": str(int(row_dict["tag_type"])),
        "lr": lr,
        "lq": lq,
        "tl": np.log1p(row_dict["text_len"]),
        "dd": row_dict["dom_depth"],
        "pt": str(int(row_dict.get("parent_tag_type", 0))),
        "sa": str(int(row_dict.get("semantic_ancestor", 0))),
    }


def main():
    print("Loading data...")
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

    # Step 1: Get well-calibrated OOF probabilities with StratifiedKFold
    print("Getting OOF GBM probabilities (StratifiedKFold)...")
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
        print(f"  Fold {fold+1}: stopped at {model.best_iteration} rounds, "
              f"loss={model.best_score['valid_0']['binary_logloss']:.5f}")

    # Raw GBM baseline
    gbm_preds = (oof_probs > 0.5).astype(int)
    print("\n=== Raw GBM OOF ===")
    print(classification_report(y, gbm_preds, target_names=["DISCARD", "KEEP"], digits=4))

    # Step 2: Build page sequences for CRF
    print("Building CRF sequences...")
    page_ids = infer_page_boundaries(df["position"].values)
    n_pages = page_ids.max() + 1

    sequences_X = []
    sequences_y = []
    for pid in range(n_pages):
        mask = page_ids == pid
        if mask.sum() < 2:
            continue
        page_df = df[mask]
        page_probs = oof_probs[mask]
        page_labels = y[mask]

        seq_x = []
        seq_y = []
        for (_, row), prob, label in zip(page_df.iterrows(), page_probs, page_labels):
            seq_x.append(block_features_for_crf(row.to_dict(), prob))
            seq_y.append(str(int(label)))
        sequences_X.append(seq_x)
        sequences_y.append(seq_y)

    print(f"  {len(sequences_X)} sequences from {n_pages} pages")

    # Step 3: Train/eval CRF with page-level split
    split = int(len(sequences_X) * 0.8)
    train_X, test_X = sequences_X[:split], sequences_X[split:]
    train_y, test_y = sequences_y[:split], sequences_y[split:]

    print(f"\nTraining CRF: {len(train_X)} train, {len(test_X)} test sequences...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.05,
        c2=0.05,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(train_X, train_y)

    # CRF predictions on test sequences
    crf_pred_seqs = crf.predict(test_X)
    crf_preds_flat = [int(l) for seq in crf_pred_seqs for l in seq]
    true_flat = [int(l) for seq in test_y for l in seq]

    # Raw GBM on same test blocks
    gbm_on_test = []
    for seq in test_X:
        for block in seq:
            gbm_on_test.append(1 if block["gbm"] > 0.5 else 0)

    print("\n=== CRF on test pages ===")
    print(classification_report(true_flat, crf_preds_flat, target_names=["DISCARD", "KEEP"], digits=4))

    print("=== Raw GBM on same test pages ===")
    print(classification_report(true_flat, gbm_on_test, target_names=["DISCARD", "KEEP"], digits=4))

    # Analyze: how many blocks did CRF flip?
    flipped = sum(1 for c, g in zip(crf_preds_flat, gbm_on_test) if c != g)
    crf_correct_flips = sum(1 for c, g, t in zip(crf_preds_flat, gbm_on_test, true_flat) if c != g and c == t)
    crf_wrong_flips = flipped - crf_correct_flips
    print(f"\nCRF flipped {flipped} blocks ({flipped/len(true_flat)*100:.1f}%)")
    print(f"  Correct flips: {crf_correct_flips}")
    print(f"  Wrong flips:   {crf_wrong_flips}")
    print(f"  Net improvement: {crf_correct_flips - crf_wrong_flips} blocks")

    # Transition weights
    print("\nTransition weights:")
    try:
        for (f, t), w in sorted(crf.transition_features_.items(), key=lambda x: -x[1]):
            fn = "KEEP" if f == "1" else "DISC"
            tn = "KEEP" if t == "1" else "DISC"
            print(f"  {fn:>4} -> {tn:<4}: {w:+.3f}")
    except Exception:
        pass

    # Top state features for KEEP
    print("\nTop 10 state features (KEEP):")
    try:
        keep_feats = {k: v for k, v in crf.state_features_.items() if k[0] == "1"}
        for (_, feat), w in sorted(keep_feats.items(), key=lambda x: -abs(x[1]))[:10]:
            print(f"  {feat:<30} {w:+.4f}")
    except Exception:
        pass

    print("\nTop 10 state features (DISCARD):")
    try:
        disc_feats = {k: v for k, v in crf.state_features_.items() if k[0] == "0"}
        for (_, feat), w in sorted(disc_feats.items(), key=lambda x: -abs(x[1]))[:10]:
            print(f"  {feat:<30} {w:+.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
