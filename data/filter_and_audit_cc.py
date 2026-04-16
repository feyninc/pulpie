"""Filter CC labeled data and prepare audit samples.

Filters:
1. Remove tiny pages (<5 blocks) — too few blocks to learn from
2. Remove all-main pages (>90% main) — almost no negative examples
3. Cap empty pages (n_main=0) at 500 — enough negative examples without overwhelming

Prepares 20 random samples for LLM audit by re-running MinerU-HTML
simplification to extract block texts alongside DeepSeek's labels.
"""

import json
import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled.jsonl')
SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
FILTERED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
AUDIT_PATH = os.path.join(SCRIPT_DIR, 'cc_audit_samples.json')

# ── MinerU-HTML module loading ──
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod

_make_module('mineru_html')
_c = _make_module('mineru_html.constants')
_c.ITEM_ID_ATTR = '_item_id'
_c.TAIL_BLOCK_TAG = 'cc-alg-uc-text'
_c.SELECT_ATTR = 'cc-select'
_c.CLASS_ATTR = 'mark-selected'

class TagType(Enum):
    Main = 'main'
    Other = 'other'
_c.TagType = TagType

_e = _make_module('mineru_html.exceptions')
class MinerUHTMLError(Exception): pass
for cn in ['MinerUHTMLPreprocessError', 'MinerUHTMLPromptError',
           'MinerUHTMLResponseParseError', 'MinerUHTMLMapToMainError',
           'MinerUHTMLFallbackError']:
    setattr(_e, cn, type(cn, (MinerUHTMLError,), {}))
_e.MinerUHTMLError = MinerUHTMLError

_b = _make_module('mineru_html.base')
@dataclass
class MinerUHTMLProcessData:
    simpled_html: str = ''
    map_html: str = ''
@dataclass
class MinerUHTMLGenerateInput:
    full_prompt: str = ''
@dataclass
class MinerUHTMLParseResult:
    item_label: dict = field(default_factory=dict)
@dataclass
class MinerUHTMLOutput:
    main_html: str = ''
@dataclass
class MinerUHTMLInput:
    raw_html: str = ''
@dataclass
class MinerUHTMLCase:
    case_id: str = ''
    input_data: MinerUHTMLInput = field(default_factory=MinerUHTMLInput)
    process_data: MinerUHTMLProcessData = field(default_factory=MinerUHTMLProcessData)
    generate_input: MinerUHTMLGenerateInput = field(default_factory=MinerUHTMLGenerateInput)
    generate_output: Optional[object] = None
    parse_result: MinerUHTMLParseResult = field(default_factory=MinerUHTMLParseResult)
    output_data: MinerUHTMLOutput = field(default_factory=MinerUHTMLOutput)
for cls in [MinerUHTMLCase, MinerUHTMLProcessData, MinerUHTMLGenerateInput,
            MinerUHTMLParseResult, MinerUHTMLOutput, MinerUHTMLInput]:
    setattr(_b, cls.__name__, cls)

_make_module('mineru_html.process')

def _load_file(mod_name, filename):
    path = os.path.join(MINERU_PATH, 'mineru_html', 'process', filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_file('mineru_html.process.html_utils', 'html_utils.py')
_simplify = _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
simplify_html = _simplify.simplify_html

import re

SEED = 42
random.seed(SEED)

MIN_BLOCKS = 5
MAX_MAIN_PCT = 0.90
MAX_EMPTY_PAGES = 500
AUDIT_SAMPLES = 20


def extract_blocks_from_simplified(simplified_html):
    """Extract block texts with their _item_id from simplified HTML."""
    blocks = {}
    pattern = re.compile(r'_item_id="(\d+)"')
    # Parse simplified HTML to get block texts
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(simplified_html, 'html.parser')
    for el in soup.find_all(attrs={'_item_id': True}):
        item_id = el.get('_item_id')
        text = el.get_text().strip()
        if text:
            blocks[item_id] = text[:300]  # truncate for audit readability
    return blocks


def main():
    # ── Load labeled data ──
    print("Loading labeled data...", flush=True)
    labeled = {}
    with open(LABELED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok':
                labeled[r['url']] = r

    print(f"  Total OK: {len(labeled)}")

    # ── Filter ──
    kept = []
    removed = {'tiny': 0, 'all_main': 0, 'empty_cap': 0}
    empty_count = 0

    for url, r in labeled.items():
        n_total = r['n_total']
        n_main = r['n_main']
        main_pct = n_main / max(n_total, 1)

        # Filter tiny pages
        if n_total < MIN_BLOCKS:
            removed['tiny'] += 1
            continue

        # Filter all-main pages
        if main_pct > MAX_MAIN_PCT and n_main > 0:
            removed['all_main'] += 1
            continue

        # Cap empty pages
        if n_main == 0:
            empty_count += 1
            if empty_count > MAX_EMPTY_PAGES:
                removed['empty_cap'] += 1
                continue

        kept.append(r)

    print(f"\n  Filtering results:")
    print(f"    Kept:         {len(kept)}")
    print(f"    Removed tiny: {removed['tiny']}")
    print(f"    Removed all-main: {removed['all_main']}")
    print(f"    Removed empty (capped): {removed['empty_cap']}")

    # Save filtered
    with open(FILTERED_PATH, 'w') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  Saved to {FILTERED_PATH}")

    # Stats on filtered set
    total_blocks = sum(r['n_total'] for r in kept)
    total_main = sum(r['n_main'] for r in kept)
    empty_kept = sum(1 for r in kept if r['n_main'] == 0)
    print(f"\n  Filtered dataset:")
    print(f"    Pages: {len(kept)}")
    print(f"    Total blocks: {total_blocks:,}")
    print(f"    Main blocks: {total_main:,} ({total_main/total_blocks*100:.1f}%)")
    print(f"    Discard blocks: {total_blocks-total_main:,} ({(total_blocks-total_main)/total_blocks*100:.1f}%)")
    print(f"    Empty pages: {empty_kept}")

    # ── Prepare audit samples ──
    print(f"\nPreparing {AUDIT_SAMPLES} audit samples...", flush=True)

    # Load HTML for sampled pages
    print("  Loading cc_sampled.jsonl index...", flush=True)
    html_by_url = {}
    with open(SAMPLED_PATH) as f:
        for line in f:
            r = json.loads(line)
            html_by_url[r['url']] = r['html']

    # Sample from non-empty kept pages (more interesting to audit)
    non_empty_kept = [r for r in kept if r['n_main'] > 0]
    audit_pool = random.sample(non_empty_kept, min(AUDIT_SAMPLES, len(non_empty_kept)))

    audit_samples = []
    for r in audit_pool:
        url = r['url']
        html = html_by_url.get(url)
        if not html:
            continue

        try:
            simplified, _ = simplify_html(html)
        except Exception:
            continue

        block_texts = extract_blocks_from_simplified(simplified)
        labels = r.get('labels', {})

        # Build audit-friendly view
        blocks_view = []
        for item_id, text in sorted(block_texts.items(), key=lambda x: int(x[0])):
            label = labels.get(item_id, 'unknown')
            blocks_view.append({
                'id': item_id,
                'label': label,
                'text': text,
            })

        audit_samples.append({
            'url': url,
            'domain': r['domain'],
            'n_main': r['n_main'],
            'n_total': r['n_total'],
            'blocks': blocks_view,
        })

    with open(AUDIT_PATH, 'w') as f:
        json.dump(audit_samples, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(audit_samples)} audit samples to {AUDIT_PATH}")
    for s in audit_samples:
        n_main = sum(1 for b in s['blocks'] if b['label'] == 'main')
        print(f"    {s['domain']:<40} {n_main}/{len(s['blocks'])} main")


if __name__ == '__main__':
    main()
