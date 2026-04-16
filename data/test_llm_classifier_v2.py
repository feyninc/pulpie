"""Test v2: LLM block classifier with richer context.

Give the LLM:
- Block text with HTML tag
- Parent tag chain (e.g., nav > ul > li)
- Position on page (early/middle/late)
- Surrounding blocks with their tags
- Page URL for context
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
SEED = 123
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
            {"role": "system", "content": "You are a web content extraction expert. You classify HTML blocks as CONTENT or BOILERPLATE based on their text, HTML structure, and position. Answer directly, no reasoning."},
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


def get_ancestor_chain(element, max_depth=5):
    """Get parent tag chain like 'article > div.content > p'."""
    chain = []
    node = element
    for _ in range(max_depth + 1):
        if not isinstance(node, Tag):
            break
        tag = node.name
        if tag in ("[document]",):
            break
        cls = node.get("class", [])
        if isinstance(cls, list):
            cls = " ".join(cls)
        # Pick most informative class token
        cls_short = ""
        if cls:
            tokens = cls.split()
            for t in tokens:
                if any(k in t.lower() for k in ["content", "article", "nav", "footer", "sidebar",
                                                   "header", "main", "post", "comment", "menu",
                                                   "widget", "social", "share", "related", "ad"]):
                    cls_short = t
                    break
            if not cls_short and tokens:
                cls_short = tokens[0]

        if cls_short:
            chain.append(f"<{tag}.{cls_short[:20]}>")
        else:
            chain.append(f"<{tag}>")
        node = node.parent

    chain.reverse()
    return " > ".join(chain)


def has_cc_select(element):
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def walk_dom_rich(element, blocks, total_blocks_estimate=0):
    """Walk DOM and collect blocks with rich metadata."""
    if not isinstance(element, Tag):
        return
    tag = element.name
    if tag in BLOCK_TAGS:
        text = element.get_text().strip()
        if len(text) >= 5:
            label = 1 if has_cc_select(element) else 0
            ancestor_chain = get_ancestor_chain(element)
            # Count links in this block
            links = element.find_all("a")
            link_text = sum(len(a.get_text()) for a in links)
            blocks.append({
                "tag": tag,
                "text": text,
                "label": label,
                "ancestors": ancestor_chain,
                "link_count": len(links),
                "link_text_len": link_text,
                "element": element,
            })
        return
    if tag in CONTAINER_TAGS or tag not in BLOCK_TAGS:
        has_block = any(isinstance(d, Tag) and d.name in BLOCK_TAGS for d in element.descendants)
        if has_block:
            for child in element.children:
                if isinstance(child, Tag):
                    walk_dom_rich(child, blocks)
        else:
            text = element.get_text().strip()
            if len(text) >= 5:
                label = 1 if has_cc_select(element) else 0
                ancestor_chain = get_ancestor_chain(element)
                links = element.find_all("a")
                link_text = sum(len(a.get_text()) for a in links)
                blocks.append({
                    "tag": tag,
                    "text": text,
                    "label": label,
                    "ancestors": ancestor_chain,
                    "link_count": len(links),
                    "link_text_len": link_text,
                    "element": element,
                })


def build_rich_prompt(url, blocks_with_context, total_blocks):
    """Build prompt with HTML structure context."""
    lines = [
        "Classify each block as CONTENT or BOILERPLATE.",
        "CONTENT = main article text, data tables, product descriptions — what the user came to read.",
        "BOILERPLATE = navigation, footer, sidebar, ads, cookie notices, sharing buttons, related links, comment forms, site chrome.",
        "",
        f"Page URL: {url}",
        f"Page has {total_blocks} total blocks.",
        "",
        "Reply with one line per block: number followed by CONTENT or BOILERPLATE.",
        "",
    ]

    for i, b in enumerate(blocks_with_context, 1):
        position_pct = b["position_pct"]
        if position_pct < 15:
            pos_label = "top of page"
        elif position_pct > 85:
            pos_label = "bottom of page"
        else:
            pos_label = f"{position_pct}% down"

        lines.append(f"--- Block {i} ---")
        lines.append(f"Position: {pos_label}")
        lines.append(f"HTML path: {b['ancestors']}")

        # Show link info if relevant
        if b["link_count"] > 0:
            text_len = len(b["text"])
            lr = b["link_text_len"] / max(text_len, 1)
            lines.append(f"Links: {b['link_count']} links, {lr:.0%} of text is linked")

        # Show surrounding context with tags
        if b["prev"]:
            ptag = b["prev"]["tag"]
            lines.append(f"[before <{ptag}>]: {b['prev']['text'][:100]}")

        lines.append(f"[text <{b['tag']}>]: {b['text'][:300]}")

        if b["next"]:
            ntag = b["next"]["tag"]
            lines.append(f"[after <{ntag}>]: {b['next']['text'][:100]}")

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

    total_blocks_tested = 0
    correct = 0
    tp = fp = tn = fn = 0
    results_by_len = {"short": [0, 0], "medium": [0, 0], "long": [0, 0]}
    results_by_label = {"content": [0, 0], "boilerplate": [0, 0]}
    pages_done = 0

    for page_idx in indices:
        if pages_done >= N_PAGES:
            break

        rec = json.loads(all_lines[page_idx])
        html = rec.get("html", "")
        url = rec.get("url", "")
        if not html or len(html) < 500:
            continue

        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        blocks = []
        walk_dom_rich(body, blocks)

        if len(blocks) < 5:
            continue

        total = len(blocks)

        # Sample up to 15 random blocks
        sample_size = min(15, total)
        sample_indices = sorted(random.sample(range(total), sample_size))

        blocks_with_context = []
        true_labels = []
        for bi in sample_indices:
            b = blocks[bi]
            prev_b = blocks[bi - 1] if bi > 0 else None
            next_b = blocks[bi + 1] if bi + 1 < total else None

            blocks_with_context.append({
                "text": b["text"],
                "tag": b["tag"],
                "ancestors": b["ancestors"],
                "link_count": b["link_count"],
                "link_text_len": b["link_text_len"],
                "position_pct": int(bi / total * 100),
                "prev": {"tag": prev_b["tag"], "text": prev_b["text"]} if prev_b else None,
                "next": {"tag": next_b["tag"], "text": next_b["text"]} if next_b else None,
            })
            true_labels.append(b["label"])

        prompt = build_rich_prompt(url, blocks_with_context, total)

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
                continue

            pred = judgments[block_idx]
            true = true_labels[j]
            text = blocks_with_context[j]["text"]
            text_len = len(text)

            total_blocks_tested += 1
            page_total += 1

            if pred == true:
                correct += 1
                page_correct += 1

            if pred == 1 and true == 1: tp += 1
            elif pred == 1 and true == 0: fp += 1
            elif pred == 0 and true == 0: tn += 1
            elif pred == 0 and true == 1: fn += 1

            cat = "short" if text_len < 30 else ("medium" if text_len < 200 else "long")
            results_by_len[cat][1] += 1
            if pred == true:
                results_by_len[cat][0] += 1

            label_cat = "content" if true == 1 else "boilerplate"
            results_by_label[label_cat][1] += 1
            if pred == true:
                results_by_label[label_cat][0] += 1

        pages_done += 1
        acc = page_correct / max(page_total, 1)
        url_short = url[:50]
        print(f"  [{pages_done:>2}/{N_PAGES}] {page_correct}/{page_total} ({acc:.0%}) | {url_short}", flush=True)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"LLM v2 (RICH CONTEXT) RESULTS ({total_blocks_tested} blocks, {pages_done} pages)")
    print(f"{'=' * 80}")

    acc = correct / max(total_blocks_tested, 1)
    print(f"\n  Overall accuracy: {correct}/{total_blocks_tested} ({acc:.1%})")

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
        c, t = results_by_label[cat]
        if t > 0:
            print(f"    {cat:<12}: {c}/{t} ({c/t:.1%})")

    print(f"\n  Comparison with v1 (text-only, same seed/pages):")
    print(f"    v1: 93.3% accuracy (527/565)")
    print(f"    v2: {acc:.1%} accuracy ({correct}/{total_blocks_tested})")


if __name__ == "__main__":
    main()
