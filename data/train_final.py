"""Train final model with best tuning config (leaves127)."""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model_dom.txt")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")


def main():
    df = pd.read_csv(TRAIN_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    X = df[feature_cols].values
    y = df["label"].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Best config from tuning: leaves127, loss=0.06925, rounds=3083
    best_config = {
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
        "scale_pos_weight": scale_pos_weight,
    }
    best_rounds = 3083

    # Held-out evaluation (single fold)
    print("Held-out fold evaluation:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain_fold = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dval_fold = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain_fold)
        model_fold = lgb.train(
            best_config, dtrain_fold, num_boost_round=best_rounds,
            valid_sets=[dval_fold],
            callbacks=[lgb.log_evaluation(0)],
        )
        y_val_pred = (model_fold.predict(X_val) > 0.5).astype(int)
        print(classification_report(y_val, y_val_pred, target_names=["DISCARD", "KEEP"]))
        break

    # Train final model on ALL data
    print(f"Training final model with {best_rounds} rounds on all data...")
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model = lgb.train(best_config, dtrain, num_boost_round=best_rounds)

    # Training set sanity check
    y_pred = (model.predict(X) > 0.5).astype(int)
    print("Training set:")
    print(classification_report(y, y_pred, target_names=["DISCARD", "KEEP"]))

    # Save
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Update features json
    meta["best_config"] = {k: v for k, v in best_config.items() if k != "verbosity"}
    meta["best_rounds"] = best_rounds
    meta["best_cv_loss"] = 0.06925
    with open(FEATURES_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {FEATURES_PATH}")


if __name__ == "__main__":
    main()
