"""Test: does the LLM correct GBM errors? Run both on the same blocks."""

import json
import os
import random
import re
import subprocess

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")
HBIRD_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")

VLLM_URL = "http://localhost:8234/v1/chat/completions"
N_PAGES = 40
SEED = 777
random.seed(SEED)

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}


def query_llm(prompt, max_tokens=300):
    resp = requests.post(VLLM_URL, json={
        "model": "Qwen/Qwen3.5-27B",
        "messages": [
            {"role": "system", "content": "You are a web content classifier. Classify each block as CONTENT or BOILERPLATE. Answer directly, no reasoning."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


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
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def walk_dom(element, blocks):
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


import lightgbm as lgb

_features_path = os.path.join(DATA_DIR, "selected_features.json")
_model_path = os.path.join(DATA_DIR, "model_dom.txt")
with open(_features_path) as _f:
    _meta = json.load(_f)
_feature_cols = _meta["features"]
_gbm_model = lgb.Booster(model_file=_model_path)


def get_gbm_predictions(html):
    """Get GBM predictions via export_features + the model."""
    clean_html = strip_annotations(html)
    rust_blocks = run_export_features(clean_html)
    if not rust_blocks:
        return []

    feature_cols = _feature_cols
    model = _gbm_model

    results = []
    for rb in rust_blocks:
        feats = []
        for f in feature_cols:
            v = rb["features"].get(f, 0)
            if isinstance(v, bool):
                v = float(v)
            elif isinstance(v, str):
                v = float(v == "true" or v == "True")
            feats.append(float(v))
        prob = model.predict([feats])[0]
        results.append({
            "text": rb["text"],
            "norm": normalize(rb["text"]),
            "gbm_prob": float(prob),
            "gbm_pred": 1 if prob > 0.5 else 0,
        })
    return results


def build_batch_prompt(blocks_with_context):
    lines = [
        "For each numbered block below, classify it as CONTENT (main article/page content that a reader came to see) or BOILERPLATE (navigation, footer, sidebar, ads, UI elements, cookie notices, related links, sharing buttons, comments section headers).",
        "",
        "Reply with one line per block: the number followed by CONTENT or BOILERPLATE.",
        "Example: 1 CONTENT",
        "",
    ]
    for i, (text, prev_text, next_text, tag) in enumerate(blocks_with_context, 1):
        lines.append(f"--- Block {i} [{tag}] ---")
        if prev_text:
            lines.append(f"[before]: {prev_text[:120]}")
        lines.append(f"[text]: {text[:300]}")
        if next_text:
            lines.append(f"[after]: {next_text[:120]}")
        lines.append("")
    return "\n".join(lines)


def parse_response(response, n_blocks):
    judgments = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(?:Block\s*)?(\d+)[.:)?\s]+(CONTENT|BOILERPLATE)', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            label = m.group(2).upper()
            if 1 <= idx <= n_blocks:
                judgments[idx] = 1 if label == "CONTENT" else 0
    return judgments


def main():
    print("Loading WebMainBench...")
    with open(BENCH_PATH) as f:
        all_lines = f.readlines()

    indices = random.sample(range(len(all_lines)), min(300, len(all_lines)))

    pages_done = 0
    all_results = []  # (true_label, gbm_pred, gbm_prob, llm_pred, text_snippet)

    for page_idx in indices:
        if pages_done >= N_PAGES:
            break

        rec = json.loads(all_lines[page_idx])
        html = rec.get("html", "")
        if not html or len(html) < 500:
            continue

        # Get ground truth blocks
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        gt_blocks = []
        walk_dom(body, gt_blocks)
        if len(gt_blocks) < 5:
            continue

        # Get GBM predictions
        gbm_blocks = get_gbm_predictions(html)
        if not gbm_blocks:
            continue

        # Match GT blocks to GBM blocks by normalized text
        gbm_map = {}
        for gb in gbm_blocks:
            norm = gb["norm"]
            if norm not in gbm_map:
                gbm_map[norm] = []
            gbm_map[norm].append(gb)

        matched = []
        for i, gt in enumerate(gt_blocks):
            norm = normalize(gt["text"])
            if norm in gbm_map and gbm_map[norm]:
                gb = gbm_map[norm].pop(0)
                matched.append({
                    "idx": i,
                    "text": gt["text"],
                    "tag": gt["tag"],
                    "true_label": gt["label"],
                    "gbm_pred": gb["gbm_pred"],
                    "gbm_prob": gb["gbm_prob"],
                })

        if len(matched) < 5:
            continue

        # Sample blocks — oversample GBM errors to get good signal
        gbm_errors = [m for m in matched if m["gbm_pred"] != m["true_label"]]
        gbm_correct = [m for m in matched if m["gbm_pred"] == m["true_label"]]

        # Take all errors + some correct ones for context
        sample = gbm_errors[:10]
        n_correct_sample = min(5, len(gbm_correct))
        if n_correct_sample > 0:
            sample += random.sample(gbm_correct, n_correct_sample)
        random.shuffle(sample)

        if not sample:
            continue

        # Build LLM prompt
        blocks_with_context = []
        for s in sample:
            idx = s["idx"]
            prev_text = gt_blocks[idx - 1]["text"] if idx > 0 else None
            next_text = gt_blocks[idx + 1]["text"] if idx + 1 < len(gt_blocks) else None
            blocks_with_context.append((s["text"], prev_text, next_text, s["tag"]))

        prompt = build_batch_prompt(blocks_with_context)

        try:
            response = query_llm(prompt, max_tokens=50 + len(sample) * 20)
        except Exception as e:
            print(f"  LLM error: {e}")
            continue

        judgments = parse_response(response, len(sample))

        for j, s in enumerate(sample):
            block_idx = j + 1
            if block_idx not in judgments:
                continue
            llm_pred = judgments[block_idx]
            all_results.append((
                s["true_label"], s["gbm_pred"], s["gbm_prob"],
                llm_pred, s["text"][:100]
            ))

        pages_done += 1
        n_errors = len(gbm_errors)
        url = rec.get("url", "")[:50]
        print(f"  [{pages_done:>2}/{N_PAGES}] {len(matched)} matched, {n_errors} GBM errors, {len(sample)} sent to LLM | {url}", flush=True)

    # ========================================
    # Analysis
    # ========================================
    print(f"\n{'=' * 80}")
    print(f"ENSEMBLE ANALYSIS ({len(all_results)} blocks)")
    print(f"{'=' * 80}")

    results = np.array([(r[0], r[1], r[3]) for r in all_results])  # true, gbm, llm
    true = results[:, 0]
    gbm = results[:, 1]
    llm = results[:, 2]

    gbm_correct = (gbm == true).sum()
    llm_correct = (llm == true).sum()
    both_correct = ((gbm == true) & (llm == true)).sum()
    both_wrong = ((gbm != true) & (llm != true)).sum()
    gbm_only = ((gbm == true) & (llm != true)).sum()
    llm_only = ((gbm != true) & (llm == true)).sum()

    print(f"\n  GBM correct:  {gbm_correct}/{len(true)} ({gbm_correct/len(true):.1%})")
    print(f"  LLM correct:  {llm_correct}/{len(true)} ({llm_correct/len(true):.1%})")
    print(f"\n  Both correct: {both_correct}")
    print(f"  Both wrong:   {both_wrong}")
    print(f"  GBM right, LLM wrong: {gbm_only}")
    print(f"  GBM wrong, LLM right: {llm_only}  <-- LLM can fix these")

    gbm_errors_mask = (gbm != true)
    n_gbm_errors = gbm_errors_mask.sum()
    llm_fixes = ((gbm != true) & (llm == true)).sum()
    llm_agrees_wrong = ((gbm != true) & (llm != true)).sum()

    print(f"\n  Of {n_gbm_errors} GBM errors:")
    print(f"    LLM corrects: {llm_fixes} ({llm_fixes/max(n_gbm_errors,1):.1%})")
    print(f"    LLM also wrong: {llm_agrees_wrong} ({llm_agrees_wrong/max(n_gbm_errors,1):.1%})")

    # But also: when LLM disagrees with GBM, is LLM usually right?
    disagree = (gbm != llm)
    n_disagree = disagree.sum()
    llm_right_on_disagree = ((gbm != llm) & (llm == true)).sum()
    gbm_right_on_disagree = ((gbm != llm) & (gbm == true)).sum()

    print(f"\n  GBM and LLM disagree on {n_disagree} blocks:")
    print(f"    LLM is right: {llm_right_on_disagree} ({llm_right_on_disagree/max(n_disagree,1):.1%})")
    print(f"    GBM is right: {gbm_right_on_disagree} ({gbm_right_on_disagree/max(n_disagree,1):.1%})")

    # Break down by GBM confidence
    probs = np.array([r[2] for r in all_results])
    confidence = np.abs(probs - 0.5) * 2

    print(f"\n  By GBM confidence on GBM errors:")
    for name, lo, hi in [("low (0-0.4)", 0, 0.4), ("med (0.4-0.7)", 0.4, 0.7), ("high (0.7-1.0)", 0.7, 1.0)]:
        mask = gbm_errors_mask & (confidence >= lo) & (confidence < hi)
        n = mask.sum()
        if n > 0:
            fixed = ((mask) & (llm == true)).sum()
            print(f"    {name}: {n} errors, LLM fixes {fixed} ({fixed/n:.1%})")

    # Show examples of LLM correcting GBM
    print(f"\n  Examples — LLM corrects GBM:")
    count = 0
    for true_l, gbm_p, gbm_prob, llm_p, text in all_results:
        if gbm_p != true_l and llm_p == true_l and count < 10:
            true_str = "KEEP" if true_l == 1 else "DISC"
            gbm_str = "KEEP" if gbm_p == 1 else "DISC"
            llm_str = "KEEP" if llm_p == 1 else "DISC"
            print(f"    true={true_str} gbm={gbm_str}(p={gbm_prob:.2f}) llm={llm_str} | {text[:80]}")
            count += 1

    print(f"\n  Examples — LLM makes it worse (GBM right, LLM wrong):")
    count = 0
    for true_l, gbm_p, gbm_prob, llm_p, text in all_results:
        if gbm_p == true_l and llm_p != true_l and count < 10:
            true_str = "KEEP" if true_l == 1 else "DISC"
            gbm_str = "KEEP" if gbm_p == 1 else "DISC"
            llm_str = "KEEP" if llm_p == 1 else "DISC"
            print(f"    true={true_str} gbm={gbm_str}(p={gbm_prob:.2f}) llm={llm_str} | {text[:80]}")
            count += 1


if __name__ == "__main__":
    main()
