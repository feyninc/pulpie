"""Prototype: CRF sequence smoothing on top of GBM block scores.

Loads training data, infers page boundaries from position resets,
trains a CRF on block sequences using GBM probabilities as features,
and compares against raw GBM thresholding.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")


def infer_page_boundaries(df):
    """Detect page boundaries from position column resets."""
    positions = df["position"].values
    page_ids = np.zeros(len(df), dtype=int)
    page_id = 0
    for i in range(1, len(df)):
        # Position resets to ~0 = new page
        if positions[i] < positions[i - 1] - 0.3:
            page_id += 1
        page_ids[i] = page_id
    return page_ids


def block_to_crf_features(row, gbm_prob):
    """Convert a block + GBM probability into CRF feature dict."""
    # Quantize GBM probability
    if gbm_prob > 0.8:
        gbm_q = "high"
    elif gbm_prob > 0.5:
        gbm_q = "mid_high"
    elif gbm_prob > 0.2:
        gbm_q = "mid_low"
    else:
        gbm_q = "low"

    # Quantize position
    pos = row["position"]
    if pos < 0.1:
        pos_q = "start"
    elif pos > 0.9:
        pos_q = "end"
    elif pos < 0.3:
        pos_q = "early"
    elif pos > 0.7:
        pos_q = "late"
    else:
        pos_q = "middle"

    # Quantize link ratio
    lr = row["link_ratio"]
    if lr > 0.5:
        lr_q = "high"
    elif lr > 0.1:
        lr_q = "mid"
    else:
        lr_q = "low"

    return {
        "gbm_prob": gbm_prob,
        "gbm_q": gbm_q,
        "position": pos,
        "pos_q": pos_q,
        "tag_type": str(int(row["tag_type"])),
        "link_ratio": lr,
        "lr_q": lr_q,
        "text_len_log": np.log1p(row["text_len"]),
        "dom_depth": row["dom_depth"],
        "parent_tag_type": str(int(row.get("parent_tag_type", 0))),
        "semantic_ancestor": str(int(row.get("semantic_ancestor", 0))),
    }


def build_sequences(df, gbm_probs, page_ids):
    """Group blocks into page sequences for CRF."""
    sequences_X = []
    sequences_y = []
    seq_page_ids = []

    unique_pages = np.unique(page_ids)
    for pid in unique_pages:
        mask = page_ids == pid
        page_df = df[mask]
        page_probs = gbm_probs[mask]
        page_labels = df["label"].values[mask]

        if len(page_df) < 2:
            continue

        seq_feats = []
        seq_labels = []
        for (_, row), prob, label in zip(page_df.iterrows(), page_probs, page_labels):
            seq_feats.append(block_to_crf_features(row, prob))
            seq_labels.append(str(int(label)))

        sequences_X.append(seq_feats)
        sequences_y.append(seq_labels)
        seq_page_ids.append(pid)

    return sequences_X, sequences_y, np.array(seq_page_ids)


def main():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    X = df[feature_cols].values
    y = df["label"].values

    print("Inferring page boundaries...")
    page_ids = infer_page_boundaries(df)
    n_pages = page_ids.max() + 1
    print(f"  Found {n_pages} pages, {len(df)} blocks")

    # Train GBM with CV to get out-of-fold probabilities (avoid overfitting)
    print("\nGetting out-of-fold GBM probabilities...")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    gbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 127,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": n_neg / max(n_pos, 1),
    }

    oof_probs = np.zeros(len(df))
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=page_ids)):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_cols)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_cols, reference=dtrain)
        model = lgb.train(
            gbm_params, dtrain, num_boost_round=3083,
            valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)],
        )
        oof_probs[val_idx] = model.predict(X[val_idx])
        print(f"  Fold {fold+1}: {model.best_iteration} rounds")

    # Evaluate raw GBM thresholding
    gbm_preds = (oof_probs > 0.5).astype(int)
    print("\n=== Raw GBM (threshold 0.5) ===")
    print(classification_report(y, gbm_preds, target_names=["DISCARD", "KEEP"]))

    # Build CRF sequences
    print("Building CRF sequences...")
    all_X, all_y, seq_pids = build_sequences(df, oof_probs, page_ids)
    print(f"  {len(all_X)} sequences")

    # CRF cross-validation (group by page)
    # Split sequences into train/test
    n_seq = len(all_X)
    split = int(n_seq * 0.8)
    train_X, test_X = all_X[:split], all_X[split:]
    train_y, test_y = all_y[:split], all_y[split:]

    print(f"\nTraining CRF on {len(train_X)} sequences, testing on {len(test_X)}...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,   # L1 regularization
        c2=0.1,   # L2 regularization
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(train_X, train_y)

    # Evaluate CRF
    crf_pred_seqs = crf.predict(test_X)
    crf_preds = [label for seq in crf_pred_seqs for label in seq]
    crf_true = [label for seq in test_y for label in seq]

    print("\n=== CRF smoothed ===")
    print(classification_report(crf_true, crf_preds, target_names=["DISCARD", "KEEP"]))

    # Also evaluate raw GBM on the same test blocks for fair comparison
    # Reconstruct which blocks are in the test set
    test_block_indices = []
    for pid in seq_pids[split:]:
        mask = page_ids == pid
        test_block_indices.extend(np.where(mask)[0])

    gbm_test_preds = gbm_preds[test_block_indices]
    gbm_test_true = y[test_block_indices]

    print("=== Raw GBM on same test set ===")
    print(classification_report(gbm_test_true, gbm_test_preds, target_names=["DISCARD", "KEEP"]))

    # Show transition weights
    print("\nCRF transition weights:")
    try:
        trans = crf.transition_features_
        for (from_label, to_label), weight in sorted(trans.items(), key=lambda x: -x[1]):
            from_name = "KEEP" if from_label == "1" else "DISCARD"
            to_name = "KEEP" if to_label == "1" else "DISCARD"
            print(f"  {from_name:>8} -> {to_name:<8}: {weight:+.3f}")
    except Exception:
        pass

    # Top CRF features
    print("\nTop CRF features (KEEP):")
    try:
        state_feats = crf.state_features_
        keep_feats = {k: v for k, v in state_feats.items() if k[0] == "1"}
        for (label, feat), weight in sorted(keep_feats.items(), key=lambda x: -abs(x[1]))[:15]:
            print(f"  {feat:<35} {weight:+.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
