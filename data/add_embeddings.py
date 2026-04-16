"""Add model2vec embeddings to training data and retrain GBM.

Reads training_data_dom.csv, computes block embeddings by running
export_features (for text) on WebMainBench, appends 256 embedding dims
as features, retrains GBM.
"""

import csv
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from model2vec import StaticModel

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WEBMAINBENCH = os.path.join(DATA_DIR, "webmainbench.jsonl")
EXPORT_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")
OUTPUT_CSV = os.path.join(DATA_DIR, "training_data_emb.csv")

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}
MIN_TEXT_LEN = 5

# Reuse labeling from DOM approach
from generate_training_data_dom import (
    label_blocks_dom, strip_annotations, run_export_features,
    match_blocks, FEATURE_COLS, TAG_TYPE_MAP,
)

EMB_DIM = 256
EMB_COLS = [f"emb_{i}" for i in range(EMB_DIM)]


def main():
    print("Loading model2vec...", flush=True)
    emb_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    print(f"  Vocab: {len(emb_model.tokenizer.get_vocab())}, dim: {emb_model.dim}", flush=True)

    print(f"Reading {WEBMAINBENCH}...", flush=True)
    total_pages = 0
    total_matched = 0
    total_keep = 0
    total_discard = 0
    skipped = 0

    all_cols = FEATURE_COLS + EMB_COLS + ["label"]

    with open(WEBMAINBENCH) as fin, open(OUTPUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(all_cols)

        for line_no, line in enumerate(fin, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            html = rec.get("html", "")
            if not html or len(html) < 100:
                skipped += 1
                continue

            dom_labels = label_blocks_dom(html)
            if not dom_labels:
                skipped += 1
                continue

            clean_html = strip_annotations(html)
            rust_blocks = run_export_features(clean_html)
            if not rust_blocks:
                skipped += 1
                continue

            matched = match_blocks(rust_blocks, dom_labels)
            if not matched:
                skipped += 1
                continue

            # Compute embeddings for matched blocks
            texts = [rb["text"] for rb, _ in matched]
            embeddings = emb_model.encode(texts)

            for (rb, label), emb in zip(matched, embeddings):
                f = rb["features"]
                row = []
                for col in FEATURE_COLS:
                    val = f[col]
                    if isinstance(val, bool):
                        row.append(int(val))
                    elif col == "tag_type":
                        row.append(TAG_TYPE_MAP.get(val, 6))
                    else:
                        row.append(val)
                row.extend(emb.tolist())
                row.append(label)
                writer.writerow(row)

                if label == 1:
                    total_keep += 1
                else:
                    total_discard += 1

            total_pages += 1
            total_matched += len(matched)

            if total_pages % 100 == 0:
                print(f"  {total_pages} pages, {total_matched} matched, "
                      f"{total_keep} keep / {total_discard} discard, "
                      f"{skipped} skipped", flush=True)

    print(f"\nDone: {total_pages} pages, {total_matched} blocks", flush=True)
    print(f"  KEEP: {total_keep} ({total_keep/max(total_matched,1)*100:.1f}%)")
    print(f"  DISCARD: {total_discard} ({total_discard/max(total_matched,1)*100:.1f}%)")
    print(f"  Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
