"""Prototype: does seeing block text improve classification?

Compare approaches:
1. GBM (current) — 40 structural features only
2. TF-IDF + LogReg — text only, no structural features
3. TF-IDF + GBM — text features added to structural features
4. Small transformer — fine-tuned on block text (if TF-IDF shows promise)

Uses same OOF evaluation as the current pipeline for fair comparison.
"""

import json
import os
import re
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import StratifiedKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom_cleaned.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")
HBIRD_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def strip_annotations(html):
    html = re.sub(r'\s+cc-select="[^"]*"', '', html)
    html = re.sub(r'\s+data-anno-uid="[^"]*"', '', html)
    html = re.sub(r'\s+class="mark-selected"', '', html)
    html = re.sub(r'</?marked-text[^>]*>', '', html)
    html = re.sub(r'</?marked-tail[^>]*>', '', html)
    return html


def has_cc_select(element):
    from bs4 import Tag
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def walk_dom(element, blocks):
    from bs4 import Tag
    if not isinstance(element, Tag):
        return
    tag = element.name
    if tag in BLOCK_TAGS:
        text = element.get_text().strip()
        if len(text) >= 5:
            label = 1 if has_cc_select(element) else 0
            blocks.append({"tag": tag, "text": text, "label": label})
        return
    if tag in CONTAINER_TAGS or tag not in BLOCK_TAGS:
        has_block = any(isinstance(d, Tag) and d.name in BLOCK_TAGS for d in element.descendants)
        if has_block:
            for child in element.children:
                if isinstance(child, Tag):
                    walk_dom(child, blocks)
        else:
            text = element.get_text().strip()
            if len(text) >= 5:
                label = 1 if has_cc_select(element) else 0
                blocks.append({"tag": tag, "text": text, "label": label})


def run_export_features(html):
    result = subprocess.run(
        [HBIRD_BIN], input=html, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def extract_block_texts():
    """Extract block texts from WebMainBench, matched to training CSV rows."""
    from bs4 import BeautifulSoup

    print("Extracting block texts from WebMainBench...")
    with open(BENCH_PATH) as f:
        bench_lines = f.readlines()

    all_texts = []  # list of (norm_text, raw_text, label)
    processed = 0

    for line_no, line in enumerate(bench_lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        html = rec.get("html", "")
        if not html or len(html) < 200:
            continue

        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        dom_blocks = []
        walk_dom(body, dom_blocks)

        clean_html = strip_annotations(html)
        rust_blocks = run_export_features(clean_html)

        if not rust_blocks or not dom_blocks:
            continue

        # Match by normalized text
        label_map = {}
        for b in dom_blocks:
            norm = normalize(b["text"])
            if norm not in label_map:
                label_map[norm] = []
            label_map[norm].append(b["label"])

        for rb in rust_blocks:
            norm = normalize(rb["text"])
            if norm in label_map and label_map[norm]:
                label = label_map[norm].pop(0)
                all_texts.append((norm, rb["text"][:500], label))

        processed += 1
        if processed % 500 == 0:
            print(f"  {processed}/{len(bench_lines)} pages, {len(all_texts)} blocks", flush=True)

    print(f"  Done: {processed} pages, {len(all_texts)} blocks")
    return all_texts


def main():
    # Load structural features
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    X_struct = df[feature_cols].values
    y = df["label"].values

    print(f"  {len(df)} blocks, {y.sum()} KEEP, {len(y)-y.sum()} DISCARD")

    # Check if we have cached block texts
    texts_cache = os.path.join(DATA_DIR, "block_texts_cache.json")
    if os.path.exists(texts_cache):
        print("Loading cached block texts...")
        with open(texts_cache) as f:
            cached = json.load(f)
        texts = cached["texts"]
        text_labels = np.array(cached["labels"])
        print(f"  {len(texts)} blocks from cache")
    else:
        block_data = extract_block_texts()
        texts = [t[1] for t in block_data]
        text_labels = np.array([t[2] for t in block_data])
        # Cache
        with open(texts_cache, "w") as f:
            json.dump({"texts": texts, "labels": text_labels.tolist()}, f)
        print(f"  Cached to {texts_cache}")

    # We need texts aligned with the CSV. Since they're extracted in the same order
    # from the same bench pages, they should match. Verify sizes.
    if len(texts) != len(df):
        print(f"\n  WARNING: text count ({len(texts)}) != CSV count ({len(df)})")
        print(f"  Using the smaller set for comparison")
        n = min(len(texts), len(df))
        texts = texts[:n]
        text_labels = text_labels[:n]
        X_struct = X_struct[:n]
        y = y[:n]

    # ========================================
    # Approach 1: GBM baseline (current model)
    # ========================================
    print("\n" + "=" * 80)
    print("Approach 1: GBM with structural features (current model)")
    print("=" * 80)

    gbm_params = meta.get("best_config", {})
    gbm_params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1})
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if "scale_pos_weight" not in gbm_params:
        gbm_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    oof_gbm = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_struct, y)):
        dtrain = lgb.Dataset(X_struct[train_idx], label=y[train_idx], feature_name=feature_cols[:X_struct.shape[1]])
        dval = lgb.Dataset(X_struct[val_idx], label=y[val_idx], feature_name=feature_cols[:X_struct.shape[1]], reference=dtrain)
        model = lgb.train(
            gbm_params, dtrain, num_boost_round=3083,
            valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)],
        )
        oof_gbm[val_idx] = model.predict(X_struct[val_idx])
        print(f"  Fold {fold+1}: stopped at {model.best_iteration} rounds", flush=True)

    gbm_preds = (oof_gbm > 0.5).astype(int)
    gbm_loss = log_loss(y, oof_gbm)
    print(f"\n  Log loss: {gbm_loss:.5f}")
    print(classification_report(y, gbm_preds, target_names=["DISCARD", "KEEP"], digits=4))

    # ========================================
    # Approach 2: TF-IDF + LogReg (text only)
    # ========================================
    print("=" * 80)
    print("Approach 2: TF-IDF + Logistic Regression (text only)")
    print("=" * 80)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5, max_df=0.95,
                            sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(texts)
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    oof_lr = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y)):
        lr = LogisticRegression(C=1.0, max_iter=500, solver="saga", n_jobs=-1)
        lr.fit(X_tfidf[train_idx], y[train_idx])
        oof_lr[val_idx] = lr.predict_proba(X_tfidf[val_idx])[:, 1]
        print(f"  Fold {fold+1} done", flush=True)

    lr_preds = (oof_lr > 0.5).astype(int)
    lr_loss = log_loss(y, oof_lr)
    print(f"\n  Log loss: {lr_loss:.5f}")
    print(classification_report(y, lr_preds, target_names=["DISCARD", "KEEP"], digits=4))

    # ========================================
    # Approach 3: TF-IDF features + GBM structural features
    # ========================================
    print("=" * 80)
    print("Approach 3: GBM structural + TF-IDF text features (top 200 TF-IDF via SVD)")
    print("=" * 80)

    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=200, random_state=42)
    X_tfidf_dense = svd.fit_transform(X_tfidf)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD: {X_tfidf_dense.shape[1]} components, {explained:.1%} variance explained")

    X_combined = np.hstack([X_struct, X_tfidf_dense])
    combined_cols = feature_cols[:X_struct.shape[1]] + [f"svd_{i}" for i in range(200)]
    print(f"  Combined feature matrix: {X_combined.shape}")

    oof_combined = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y)):
        dtrain = lgb.Dataset(X_combined[train_idx], label=y[train_idx], feature_name=combined_cols)
        dval = lgb.Dataset(X_combined[val_idx], label=y[val_idx], feature_name=combined_cols, reference=dtrain)
        model = lgb.train(
            gbm_params, dtrain, num_boost_round=3083,
            valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)],
        )
        oof_combined[val_idx] = model.predict(X_combined[val_idx])
        print(f"  Fold {fold+1}: stopped at {model.best_iteration} rounds", flush=True)

    combined_preds = (oof_combined > 0.5).astype(int)
    combined_loss = log_loss(y, oof_combined)
    print(f"\n  Log loss: {combined_loss:.5f}")
    print(classification_report(y, combined_preds, target_names=["DISCARD", "KEEP"], digits=4))

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Approach':<45} {'Log Loss':>10} {'Accuracy':>10} {'F1 KEEP':>10}")
    print("-" * 75)

    from sklearn.metrics import accuracy_score, f1_score

    for name, probs, preds in [
        ("GBM structural (current)", oof_gbm, gbm_preds),
        ("TF-IDF + LogReg (text only)", oof_lr, lr_preds),
        ("GBM structural + SVD text", oof_combined, combined_preds),
    ]:
        ll = log_loss(y, probs)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        print(f"{name:<45} {ll:>10.5f} {acc:>10.4f} {f1:>10.4f}")

    # Where does text help? Look at high-confidence GBM errors corrected by text
    print("\n\nHigh-confidence GBM errors where text model disagrees:")
    hc_gbm_errors = ((gbm_preds != y) & (np.abs(oof_gbm - 0.5) > 0.4))
    text_corrects = (lr_preds == y) & hc_gbm_errors
    print(f"  GBM high-confidence errors: {hc_gbm_errors.sum()}")
    print(f"  Text model corrects: {text_corrects.sum()} ({text_corrects.sum()/max(hc_gbm_errors.sum(),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
