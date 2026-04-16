"""Audit suspected noisy labels using Qwen3.5-27B via vLLM.

Picks pages with high-confidence disagreements, extracts blocks with text,
and asks the LLM to judge each suspicious block with actual text context.
"""

import json
import os
import random
import re
import subprocess
import tempfile

import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")
HBIRD_BIN = os.path.join(DATA_DIR, "..", "target", "release", "export_features")

VLLM_URL = "http://localhost:8234/v1/chat/completions"
N_PAGES = 30
N_BLOCKS_PER_PAGE = 3  # max suspicious blocks to audit per page
random.seed(42)

# Must match segment.rs
BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}


def query_llm(prompt, max_tokens=100):
    resp = requests.post(VLLM_URL, json={
        "model": "Qwen/Qwen3.5-27B",
        "messages": [
            {"role": "system", "content": "You are a web content auditor. Answer directly and concisely. No chain of thought."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


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


def walk_dom_labeled(element, blocks):
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
        has_block = any(
            isinstance(d, Tag) and d.name in BLOCK_TAGS
            for d in element.descendants
        )
        if has_block:
            for child in element.children:
                if isinstance(child, Tag):
                    walk_dom_labeled(child, blocks)
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


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def process_page(html):
    """Get labeled blocks with text + GBM probabilities."""
    from bs4 import BeautifulSoup

    # Get labels from annotated HTML
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    labeled_blocks = []
    walk_dom_labeled(body, labeled_blocks)

    # Get features from clean HTML
    clean_html = strip_annotations(html)
    rust_blocks = run_export_features(clean_html)
    if not rust_blocks:
        return []

    # Match by normalized text
    label_map = {}
    for b in labeled_blocks:
        norm = normalize(b["text"])
        if norm not in label_map:
            label_map[norm] = []
        label_map[norm].append(b["label"])

    matched = []
    for rb in rust_blocks:
        norm = normalize(rb["text"])
        if norm in label_map and label_map[norm]:
            label = label_map[norm].pop(0)
            matched.append({
                "text": rb["text"][:500],  # truncate for prompt
                "tag": rb["features"]["tag_type"],
                "label": label,
                "prob": rb["features"].get("_prob", 0.5),  # not available, will use features
                "features": rb["features"],
            })

    return matched


def build_prompt(block, prev_text, next_text):
    """Ask the LLM to judge a block with actual text context."""
    label_str = "KEEP (main content)" if block["label"] == 1 else "DISCARD (boilerplate)"
    text = block["text"][:400]

    prompt = f"""You are auditing labels in a web content extraction dataset. Each HTML block was labeled as either KEEP (main page content) or DISCARD (boilerplate/navigation/ads/footer).

Here is the block text and its surrounding context:

PREVIOUS BLOCK: {prev_text[:200] if prev_text else "(start of page)"}

>>> THIS BLOCK (labeled {label_str}): {text} <<<

NEXT BLOCK: {next_text[:200] if next_text else "(end of page)"}

Based on the text content and context, is this block truly main article content or boilerplate?

Reply ONLY with one line in this format:
CORRECT: [reason] — if the label is right
ERROR: [reason] — if this looks mislabeled
Do not include any other text or explanation."""

    return prompt


def main():
    from bs4 import BeautifulSoup

    print("Loading WebMainBench pages...")
    with open(BENCH_PATH) as f:
        all_lines = f.readlines()

    # Sample pages
    indices = random.sample(range(len(all_lines)), min(200, len(all_lines)))

    audited = 0
    results = {"correct": 0, "error": 0, "uncertain": 0}
    results_by_type = {"A": {"correct": 0, "error": 0, "uncertain": 0},
                       "B": {"correct": 0, "error": 0, "uncertain": 0}}
    details = []

    for page_idx in indices:
        if audited >= 50:
            break

        rec = json.loads(all_lines[page_idx])
        html = rec.get("html", "")
        if not html or len(html) < 200:
            continue

        blocks = process_page(html)
        if len(blocks) < 5:
            continue

        # Find suspicious blocks: use simple heuristics since we don't have GBM probs
        # A block is suspicious if it looks like content but labeled DISCARD, or vice versa
        suspicious = []
        for i, b in enumerate(blocks):
            text = b["text"].strip()
            link_ratio = b["features"].get("link_ratio", 0)
            text_len = b["features"].get("text_len", 0)

            # Type A: labeled DISCARD but looks like content (long text, low link ratio)
            if b["label"] == 0 and text_len > 100 and link_ratio < 0.1:
                suspicious.append((i, "A"))
            # Type B: labeled KEEP but looks like boilerplate (short, high link ratio, or nav-like)
            elif b["label"] == 1 and (link_ratio > 0.5 or text_len < 20):
                suspicious.append((i, "B"))

        if not suspicious:
            continue

        # Take up to N_BLOCKS_PER_PAGE
        to_audit = suspicious[:N_BLOCKS_PER_PAGE]

        for block_i, stype in to_audit:
            if audited >= 50:
                break

            block = blocks[block_i]
            prev_text = blocks[block_i - 1]["text"] if block_i > 0 else None
            next_text = blocks[block_i + 1]["text"] if block_i + 1 < len(blocks) else None

            prompt = build_prompt(block, prev_text, next_text)
            try:
                response = query_llm(prompt)
            except Exception as e:
                print(f"  LLM error: {e}")
                continue

            resp_upper = response.upper()
            if "ERROR" in resp_upper.split("\n")[0]:
                verdict = "error"
            elif "CORRECT" in resp_upper.split("\n")[0]:
                verdict = "correct"
            else:
                verdict = "uncertain"

            results[verdict] += 1
            results_by_type[stype][verdict] += 1
            audited += 1

            label_str = "KEEP" if block["label"] == 1 else "DISC"
            print(f"  [{audited:>2}/50] type={stype} label={label_str:<4} "
                  f"len={len(block['text']):>5} lr={block['features'].get('link_ratio',0):.2f} "
                  f"-> {verdict.upper()}: {response[:100]}", flush=True)

            details.append({
                "url": rec.get("url", ""),
                "type": stype,
                "label": "KEEP" if block["label"] == 1 else "DISCARD",
                "text": block["text"][:300],
                "verdict": verdict,
                "response": response[:300],
            })

    # Summary
    total = sum(results.values())
    print(f"\n{'='*80}")
    print(f"AUDIT RESULTS ({total} blocks)")
    print(f"{'='*80}")
    print(f"  CORRECT labels:  {results['correct']:>3} ({results['correct']/max(total,1)*100:.0f}%)")
    print(f"  LABELING ERRORS: {results['error']:>3} ({results['error']/max(total,1)*100:.0f}%)")
    print(f"  UNCERTAIN:       {results['uncertain']:>3} ({results['uncertain']/max(total,1)*100:.0f}%)")

    print(f"\nBy type:")
    for t in ["A", "B"]:
        r = results_by_type[t]
        tt = sum(r.values())
        if tt == 0:
            continue
        desc = "labeled DISCARD but looks like content" if t == "A" else "labeled KEEP but looks like boilerplate"
        print(f"  Type {t} ({desc}):")
        print(f"    CORRECT: {r['correct']}/{tt}  ERROR: {r['error']}/{tt}  UNCERTAIN: {r['uncertain']}/{tt}")

    out_path = os.path.join(DATA_DIR, "label_audit_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": results, "by_type": results_by_type, "details": details}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
