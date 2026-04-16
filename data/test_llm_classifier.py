"""Test: can Qwen3.5-27B classify blocks as content/boilerplate from text alone?

Sample random blocks from WebMainBench, send to LLM with surrounding context,
compare against ground truth cc-select labels. This measures the ceiling
for any text-based classifier approach.
"""

import json
import os
import random
import re
import subprocess

import requests
from bs4 import BeautifulSoup, Tag

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")
HBIRD_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")

VLLM_URL = "http://localhost:8234/v1/chat/completions"
N_PAGES = 40
SEED = 123  # different from training seed
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


def build_batch_prompt(blocks_with_context):
    """Build a prompt for classifying multiple blocks on one page."""
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

    indices = random.sample(range(len(all_lines)), min(200, len(all_lines)))

    total_blocks = 0
    correct = 0
    wrong = 0
    skipped = 0
    pages_done = 0

    # Per-class stats
    tp = fp = tn = fn = 0

    # By block characteristics
    results_by_len = {"short": [0, 0], "medium": [0, 0], "long": [0, 0]}  # [correct, total]
    results_by_true_label = {"content": [0, 0], "boilerplate": [0, 0]}

    for page_idx in indices:
        if pages_done >= N_PAGES:
            break

        rec = json.loads(all_lines[page_idx])
        html = rec.get("html", "")
        if not html or len(html) < 500:
            continue

        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        blocks = []
        walk_dom(body, blocks)

        if len(blocks) < 5:
            continue

        # Sample up to 15 random blocks from this page (mix of content and boilerplate)
        sample_size = min(15, len(blocks))
        sample_indices = sorted(random.sample(range(len(blocks)), sample_size))

        blocks_with_context = []
        true_labels = []
        for bi in sample_indices:
            b = blocks[bi]
            prev_text = blocks[bi - 1]["text"] if bi > 0 else None
            next_text = blocks[bi + 1]["text"] if bi + 1 < len(blocks) else None
            blocks_with_context.append((b["text"], prev_text, next_text, b["tag"]))
            true_labels.append(b["label"])

        prompt = build_batch_prompt(blocks_with_context)

        try:
            response = query_llm(prompt, max_tokens=50 + sample_size * 20)
        except Exception as e:
            print(f"  LLM error: {e}")
            continue

        judgments = parse_response(response, len(blocks_with_context))

        page_correct = 0
        page_total = 0
        for j in range(len(blocks_with_context)):
            block_idx = j + 1
            if block_idx not in judgments:
                skipped += 1
                continue

            pred = judgments[block_idx]
            true = true_labels[j]
            text = blocks_with_context[j][0]
            text_len = len(text)

            total_blocks += 1
            page_total += 1

            if pred == true:
                correct += 1
                page_correct += 1
            else:
                wrong += 1

            # Confusion matrix
            if pred == 1 and true == 1: tp += 1
            elif pred == 1 and true == 0: fp += 1
            elif pred == 0 and true == 0: tn += 1
            elif pred == 0 and true == 1: fn += 1

            # By length
            if text_len < 30:
                cat = "short"
            elif text_len < 200:
                cat = "medium"
            else:
                cat = "long"
            results_by_len[cat][1] += 1
            if pred == true:
                results_by_len[cat][0] += 1

            # By true label
            label_cat = "content" if true == 1 else "boilerplate"
            results_by_true_label[label_cat][1] += 1
            if pred == true:
                results_by_true_label[label_cat][0] += 1

        pages_done += 1
        acc = page_correct / max(page_total, 1)
        url = rec.get("url", "")[:50]
        print(f"  [{pages_done:>2}/{N_PAGES}] {page_correct}/{page_total} correct ({acc:.0%}) | {url}", flush=True)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"LLM BLOCK CLASSIFIER RESULTS ({total_blocks} blocks from {pages_done} pages)")
    print(f"{'=' * 80}")

    acc = correct / max(total_blocks, 1)
    print(f"\n  Overall accuracy: {correct}/{total_blocks} ({acc:.1%})")
    print(f"  Skipped (unparseable response): {skipped}")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"\n  Content (KEEP):")
    print(f"    Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    precision_d = tn / max(tn + fn, 1)
    recall_d = tn / max(tn + fp, 1)
    f1_d = 2 * precision_d * recall_d / max(precision_d + recall_d, 1e-9)
    print(f"  Boilerplate (DISCARD):")
    print(f"    Precision: {precision_d:.4f}  Recall: {recall_d:.4f}  F1: {f1_d:.4f}")

    print(f"\n  Confusion matrix:")
    print(f"    {'':>20} Pred BOILERPLATE  Pred CONTENT")
    print(f"    {'True BOILERPLATE':>20}  {tn:>10}  {fp:>10}")
    print(f"    {'True CONTENT':>20}  {fn:>10}  {tp:>10}")

    print(f"\n  By text length:")
    for cat in ["short", "medium", "long"]:
        c, t = results_by_len[cat]
        if t > 0:
            print(f"    {cat:<10}: {c}/{t} ({c/t:.1%})")

    print(f"\n  By true label:")
    for cat in ["content", "boilerplate"]:
        c, t = results_by_true_label[cat]
        if t > 0:
            print(f"    {cat:<12}: {c}/{t} ({c/t:.1%})")


if __name__ == "__main__":
    main()
