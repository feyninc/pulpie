"""Optimize LLM prompt for block classification.

1. Run baseline prompt, collect errors
2. Feed errors to LLM, ask it to write a better prompt (OPRO-style)
3. Test new prompt
4. Repeat for a few rounds
Also test: few-shot prompting with hard examples.
"""

import json
import os
import random
import re
import copy

import requests
from bs4 import BeautifulSoup, Tag

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(DATA_DIR, "webmainbench.jsonl")

VLLM_URL = "http://localhost:8234/v1/chat/completions"
SEED = 123
random.seed(SEED)

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}


def query_llm(messages, max_tokens=300, temperature=0.0):
    resp = requests.post(VLLM_URL, json={
        "model": "Qwen/Qwen3.5-27B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def has_cc_select(element):
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def get_ancestor_chain(element, max_depth=4):
    chain = []
    node = element
    for _ in range(max_depth + 1):
        if not isinstance(node, Tag) or node.name in ("[document]",):
            break
        tag = node.name
        cls = node.get("class", [])
        if isinstance(cls, list):
            cls = " ".join(cls)
        cls_short = ""
        if cls:
            tokens = cls.split()[:2]
            cls_short = ".".join(tokens)
        chain.append(f"{tag}.{cls_short}" if cls_short else tag)
        node = node.parent
    chain.reverse()
    return " > ".join(chain)


def walk_dom(element, blocks):
    if not isinstance(element, Tag):
        return
    tag = element.name
    if tag in BLOCK_TAGS:
        text = element.get_text().strip()
        if len(text) >= 5:
            label = 1 if has_cc_select(element) else 0
            blocks.append({
                "tag": tag, "text": text, "label": label,
                "ancestors": get_ancestor_chain(element),
                "link_count": len(element.find_all("a")),
            })
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
                blocks.append({
                    "tag": tag, "text": text, "label": label,
                    "ancestors": get_ancestor_chain(element),
                    "link_count": len(element.find_all("a")),
                })


def load_pages(n_pages=60):
    """Load pages and extract blocks."""
    with open(BENCH_PATH) as f:
        all_lines = f.readlines()

    indices = random.sample(range(len(all_lines)), min(300, len(all_lines)))

    pages = []
    for page_idx in indices:
        if len(pages) >= n_pages:
            break
        rec = json.loads(all_lines[page_idx])
        html = rec.get("html", "")
        url = rec.get("url", "")
        if not html or len(html) < 500:
            continue

        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body") or soup
        blocks = []
        walk_dom(body, blocks)
        if len(blocks) < 5:
            continue

        pages.append({"url": url, "blocks": blocks})

    return pages


def build_prompt_for_blocks(system_msg, url, sample_blocks, total_blocks):
    """Build the user message for classifying blocks."""
    lines = [
        f"Page: {url}",
        f"Total blocks on page: {total_blocks}",
        "",
    ]

    for i, b in enumerate(sample_blocks, 1):
        pos_pct = b["position_pct"]
        lines.append(f"--- Block {i} ---")
        lines.append(f"Position: {pos_pct}% | HTML: {b['ancestors']} | Links: {b['link_count']}")
        if b["prev"]:
            lines.append(f"  [before <{b['prev']['tag']}>]: {b['prev']['text'][:80]}")
        lines.append(f"  [text <{b['tag']}>]: {b['text'][:250]}")
        if b["next"]:
            lines.append(f"  [after <{b['next']['tag']}>]: {b['next']['text'][:80]}")
        lines.append("")

    lines.append("Reply: one line per block, number then CONTENT or BOILERPLATE.")
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


def evaluate_prompt(system_msg, pages, max_pages=None):
    """Run the prompt on pages, return accuracy and error details."""
    correct = 0
    total = 0
    errors = []

    n = max_pages or len(pages)
    for page in pages[:n]:
        url = page["url"]
        blocks = page["blocks"]
        n_blocks = len(blocks)

        sample_size = min(15, n_blocks)
        sample_indices = sorted(random.sample(range(n_blocks), sample_size))

        sample_blocks = []
        true_labels = []
        for bi in sample_indices:
            b = blocks[bi]
            prev_b = blocks[bi - 1] if bi > 0 else None
            next_b = blocks[bi + 1] if bi + 1 < n_blocks else None
            sample_blocks.append({
                "text": b["text"], "tag": b["tag"],
                "ancestors": b["ancestors"], "link_count": b["link_count"],
                "position_pct": int(bi / n_blocks * 100),
                "prev": {"tag": prev_b["tag"], "text": prev_b["text"]} if prev_b else None,
                "next": {"tag": next_b["tag"], "text": next_b["text"]} if next_b else None,
            })
            true_labels.append(b["label"])

        user_msg = build_prompt_for_blocks(system_msg, url, sample_blocks, n_blocks)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = query_llm(messages, max_tokens=50 + sample_size * 20)
        except Exception as e:
            continue

        judgments = parse_response(response, len(sample_blocks))

        for j in range(len(sample_blocks)):
            block_idx = j + 1
            if block_idx not in judgments:
                continue

            pred = judgments[block_idx]
            true = true_labels[j]
            total += 1
            if pred == true:
                correct += 1
            else:
                errors.append({
                    "url": url[:60],
                    "text": sample_blocks[j]["text"][:150],
                    "tag": sample_blocks[j]["tag"],
                    "ancestors": sample_blocks[j]["ancestors"],
                    "position": sample_blocks[j]["position_pct"],
                    "link_count": sample_blocks[j]["link_count"],
                    "true": "CONTENT" if true == 1 else "BOILERPLATE",
                    "pred": "CONTENT" if pred == 1 else "BOILERPLATE",
                })

    acc = correct / max(total, 1)
    return acc, correct, total, errors


def optimize_prompt(current_prompt, errors, round_num):
    """Ask the LLM to improve the prompt based on errors."""
    error_examples = random.sample(errors, min(20, len(errors)))
    error_text = "\n".join([
        f"  - True={e['true']}, Predicted={e['pred']}, tag=<{e['tag']}>, "
        f"ancestors={e['ancestors']}, pos={e['position']}%, links={e['link_count']}, "
        f"text=\"{e['text'][:100]}\""
        for e in error_examples
    ])

    meta_prompt = f"""You are optimizing a system prompt for an LLM that classifies HTML blocks as CONTENT or BOILERPLATE.

Current system prompt:
---
{current_prompt}
---

The classifier made these errors (round {round_num}):
{error_text}

Total errors: {len(errors)}

Common error patterns to address:
- Blocks inside nav/footer/aside ancestors that look like content text
- Short navigational items ("Home", "About Us") being classified as content
- Structured data (tables, listings) being misclassified
- Sidebar content vs main content ambiguity

Write an improved system prompt that would reduce these errors. The prompt should be concise (under 300 words), actionable, and specific about edge cases. Output ONLY the new system prompt text, nothing else."""

    messages = [
        {"role": "system", "content": "You are a prompt engineering expert. Output only the improved prompt."},
        {"role": "user", "content": meta_prompt},
    ]
    return query_llm(messages, max_tokens=500, temperature=0.3)


def main():
    print("Loading pages...")
    all_pages = load_pages(60)
    random.shuffle(all_pages)

    # Split into dev (for optimization) and test (for final eval)
    dev_pages = all_pages[:30]
    test_pages = all_pages[30:]
    print(f"  Dev: {len(dev_pages)} pages, Test: {len(test_pages)} pages")

    # ========================================
    # Baseline prompt
    # ========================================
    baseline_system = (
        "You are a web content extraction expert. Classify each HTML block as CONTENT or BOILERPLATE.\n"
        "CONTENT = main article text, data tables, product descriptions — what the user came to read.\n"
        "BOILERPLATE = navigation, footer, sidebar, ads, cookie notices, sharing buttons, related links, comment forms, site chrome.\n"
        "Answer directly, no reasoning."
    )

    print(f"\n{'='*80}")
    print("BASELINE PROMPT (dev set)")
    print(f"{'='*80}")
    random.seed(SEED)
    acc, c, t, errors = evaluate_prompt(baseline_system, dev_pages)
    print(f"  Accuracy: {c}/{t} ({acc:.1%}), Errors: {len(errors)}")

    print(f"\n{'='*80}")
    print("BASELINE PROMPT (test set)")
    print(f"{'='*80}")
    random.seed(SEED + 1)
    test_acc_baseline, tc, tt, _ = evaluate_prompt(baseline_system, test_pages)
    print(f"  Accuracy: {tc}/{tt} ({test_acc_baseline:.1%})")

    # ========================================
    # Few-shot prompt — add examples of hard cases
    # ========================================
    fewshot_system = (
        "You are a web content extraction expert. Classify each HTML block as CONTENT or BOILERPLATE.\n\n"
        "CONTENT = the main text a visitor came to read: article body, product details, forum posts, data tables that ARE the page's purpose.\n"
        "BOILERPLATE = site chrome that appears on every page: navigation menus, footers, sidebars, ads, cookie banners, share buttons, related article links, comment form labels, breadcrumbs, login prompts.\n\n"
        "Key rules:\n"
        "- Blocks inside <nav>, <footer>, <aside> ancestors are almost always BOILERPLATE\n"
        "- Blocks at position <5% or >90% of the page are likely BOILERPLATE\n"
        "- Blocks with many links relative to text are likely BOILERPLATE (navigation)\n"
        "- Short blocks like 'Home', 'About', 'Share', 'Subscribe' are BOILERPLATE\n"
        "- Even if text looks informative, if it's in a sidebar/nav ancestor it's BOILERPLATE\n"
        "- Product specs, article paragraphs, forum reply text, table data = CONTENT\n\n"
        "Answer directly: number then CONTENT or BOILERPLATE."
    )

    print(f"\n{'='*80}")
    print("FEW-SHOT/RULES PROMPT (dev set)")
    print(f"{'='*80}")
    random.seed(SEED)
    acc_fs, c_fs, t_fs, errors_fs = evaluate_prompt(fewshot_system, dev_pages)
    print(f"  Accuracy: {c_fs}/{t_fs} ({acc_fs:.1%}), Errors: {len(errors_fs)}")

    print(f"\n{'='*80}")
    print("FEW-SHOT/RULES PROMPT (test set)")
    print(f"{'='*80}")
    random.seed(SEED + 1)
    test_acc_fs, tc_fs, tt_fs, _ = evaluate_prompt(fewshot_system, test_pages)
    print(f"  Accuracy: {tc_fs}/{tt_fs} ({test_acc_fs:.1%})")

    # ========================================
    # OPRO-style optimization (3 rounds)
    # ========================================
    print(f"\n{'='*80}")
    print("OPRO-STYLE OPTIMIZATION")
    print(f"{'='*80}")

    current_prompt = fewshot_system  # start from the better prompt
    best_prompt = current_prompt
    best_acc = acc_fs
    best_errors = errors_fs

    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        print(f"  Current dev accuracy: {best_acc:.1%} ({len(best_errors)} errors)")

        # Generate improved prompt
        new_prompt = optimize_prompt(current_prompt, best_errors, round_num)
        print(f"  Generated new prompt ({len(new_prompt)} chars)")
        print(f"  Preview: {new_prompt[:150]}...")

        # Evaluate on dev set
        random.seed(SEED)
        acc_new, c_new, t_new, errors_new = evaluate_prompt(new_prompt, dev_pages)
        print(f"  New dev accuracy: {c_new}/{t_new} ({acc_new:.1%}), Errors: {len(errors_new)}")

        if acc_new > best_acc:
            print(f"  IMPROVED! {best_acc:.1%} -> {acc_new:.1%}")
            best_acc = acc_new
            best_prompt = new_prompt
            best_errors = errors_new
            current_prompt = new_prompt
        else:
            print(f"  No improvement ({acc_new:.1%} vs {best_acc:.1%})")
            # Still use new prompt as base for next round to explore
            current_prompt = new_prompt
            best_errors = errors_new  # use latest errors for next optimization

    # ========================================
    # Final evaluation on test set
    # ========================================
    print(f"\n{'='*80}")
    print("FINAL: BEST OPTIMIZED PROMPT (test set)")
    print(f"{'='*80}")
    print(f"\nBest prompt:\n{best_prompt}\n")

    random.seed(SEED + 1)
    test_acc_opt, tc_opt, tt_opt, test_errors = evaluate_prompt(best_prompt, test_pages)
    print(f"  Test accuracy: {tc_opt}/{tt_opt} ({test_acc_opt:.1%})")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Prompt':<30} {'Dev Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline':<30} {acc:.1%}{'':>5} {test_acc_baseline:.1%}")
    print(f"  {'Few-shot/Rules':<30} {acc_fs:.1%}{'':>5} {test_acc_fs:.1%}")
    print(f"  {'OPRO-optimized':<30} {best_acc:.1%}{'':>5} {test_acc_opt:.1%}")

    # Save best prompt
    out_path = os.path.join(DATA_DIR, "best_llm_prompt.txt")
    with open(out_path, "w") as f:
        f.write(best_prompt)
    print(f"\n  Best prompt saved to {out_path}")


if __name__ == "__main__":
    main()
