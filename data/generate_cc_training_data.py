"""Generate CC training data for hummingbird GBM.

Pipeline:
1. Load Dripper double-check results, filter to pages with >=70% agreement
2. Load DeepSeek labels for those pages
3. Re-run MinerU-HTML simplification to get block texts per _item_id
4. Run hummingbird export_features on raw HTML to get Rust blocks + features
5. Match Rust blocks to MinerU-HTML blocks by normalized text, assign labels
6. Output as CSV in same format as training_data_dom.csv
"""

import csv
import json
import os
import re
import subprocess
import sys
import time

from bs4 import BeautifulSoup

# ── MinerU-HTML module loading (same as other CC scripts) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')
HUMMINGBIRD_BIN = os.path.join(SCRIPT_DIR, '..', 'target', 'release', 'export_features')

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

# ── Paths ──
SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
DOUBLECHECK_PATH = os.path.join(SCRIPT_DIR, 'cc_doublecheck_results.jsonl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'training_data_cc.csv')

# Feature columns (must match generate_training_data_dom.py / Rust Features struct)
FEATURE_COLS = [
    "text_len", "word_count", "sentence_count", "comma_count",
    "avg_word_length", "stop_word_ratio", "capitalization_ratio",
    "punctuation_density", "has_copyright", "has_date_pattern",
    "link_len", "link_count", "link_ratio", "tag_count",
    "paragraph_count", "heading_count", "list_item_count", "image_count",
    "text_to_tag_ratio",
    "class_id_score", "parent_class_id_score", "tag_type", "tag_type_score",
    "dom_depth", "position", "distance_from_end",
    "is_first_10pct", "is_last_10pct", "has_boilerplate_class",
    "parent_tag_type", "semantic_ancestor",
    "prev_block_text_len", "prev_block_link_ratio",
    "next_block_text_len", "next_block_link_ratio",
    "blocks_since_heading", "blocks_until_heading",
    "section_heading_text_len", "section_block_count", "section_link_density",
    "page_total_blocks", "page_total_text_len", "page_total_link_ratio",
    "page_heading_count", "block_text_len_ratio",
]

TAG_TYPE_MAP = {
    "Paragraph": 0, "Heading": 1, "ListItem": 2,
    "Preformatted": 3, "TableCell": 4, "Blockquote": 5, "Other": 6,
}

MIN_AGREEMENT = 0.70


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def extract_block_texts(simplified_html):
    """Extract block texts with their _item_id from simplified HTML."""
    soup = BeautifulSoup(simplified_html, 'html.parser')
    blocks = {}
    for el in soup.find_all(attrs={'_item_id': True}):
        item_id = el.get('_item_id')
        text = el.get_text()
        if text.strip():
            blocks[item_id] = text
    return blocks


def run_export_features(html):
    result = subprocess.run(
        [HUMMINGBIRD_BIN], input=html, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def feature_row(block, label):
    f = block["features"]
    row = []
    for col in FEATURE_COLS:
        val = f[col]
        if isinstance(val, bool):
            row.append(int(val))
        elif col == "tag_type":
            row.append(TAG_TYPE_MAP.get(val, 6))
        else:
            row.append(val)
    row.append(label)
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-agreement', type=float, default=MIN_AGREEMENT)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    # Step 1: Load agreement rates from double-check
    print('Loading double-check results...', flush=True)
    agreement_by_url = {}
    with open(DOUBLECHECK_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok':
                agreement_by_url[r['url']] = r['agreement_rate']
    print(f'  {len(agreement_by_url)} pages with agreement data', flush=True)

    # Step 2: Load DeepSeek labels, filter by agreement
    print(f'Loading DeepSeek labels (min agreement >= {args.min_agreement})...', flush=True)
    labels_by_url = {}
    filtered_out = 0
    no_agreement = 0
    with open(LABELED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') != 'ok':
                continue
            url = r['url']
            agr = agreement_by_url.get(url)
            if agr is None:
                no_agreement += 1
                continue
            if agr < args.min_agreement:
                filtered_out += 1
                continue
            labels_by_url[url] = r.get('labels', {})

    print(f'  Kept: {len(labels_by_url)} pages', flush=True)
    print(f'  Filtered (low agreement): {filtered_out}', flush=True)
    print(f'  No agreement data: {no_agreement}', flush=True)

    # Step 3: Load HTML
    print('Loading HTML from cc_sampled.jsonl...', flush=True)
    html_by_url = {}
    with open(SAMPLED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r['url'] in labels_by_url:
                html_by_url[r['url']] = r['html']
    print(f'  Found HTML for {len(html_by_url)} pages', flush=True)

    urls = list(html_by_url.keys())
    if args.limit > 0:
        urls = urls[:args.limit]
        print(f'  Limited to {len(urls)}', flush=True)

    # Step 4: Process each page
    print(f'\nProcessing {len(urls)} pages...', flush=True)
    t_start = time.time()
    stats = {
        'pages_ok': 0, 'simplify_fail': 0, 'features_fail': 0,
        'no_match': 0, 'total_blocks': 0, 'matched_blocks': 0,
        'keep': 0, 'discard': 0,
    }

    with open(OUTPUT_PATH, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(FEATURE_COLS + ['label'])

        for i, url in enumerate(urls):
            html = html_by_url[url]
            ds_labels = labels_by_url[url]  # item_id -> "main"/"other"

            # Simplify to get block texts by _item_id
            try:
                simplified, _ = simplify_html(html)
            except Exception:
                stats['simplify_fail'] += 1
                continue

            block_texts = extract_block_texts(simplified)
            if not block_texts:
                stats['simplify_fail'] += 1
                continue

            # Build label map: normalized_text -> label (1=keep, 0=discard)
            label_map = {}
            for item_id, text in block_texts.items():
                label_str = ds_labels.get(item_id)
                if label_str is None:
                    continue
                norm = normalize(text)
                if norm not in label_map:
                    label_map[norm] = []
                label_map[norm].append(1 if label_str == 'main' else 0)

            if not label_map:
                stats['no_match'] += 1
                continue

            # Run hummingbird feature extraction on raw HTML
            rust_blocks = run_export_features(html)
            if not rust_blocks:
                stats['features_fail'] += 1
                continue

            # Match Rust blocks to labeled blocks by text
            page_matched = 0
            for rb in rust_blocks:
                norm = normalize(rb['text'])
                if norm in label_map and label_map[norm]:
                    label = label_map[norm].pop(0)
                    writer.writerow(feature_row(rb, label))
                    page_matched += 1
                    if label == 1:
                        stats['keep'] += 1
                    else:
                        stats['discard'] += 1

            stats['total_blocks'] += len(rust_blocks)
            stats['matched_blocks'] += page_matched

            if page_matched > 0:
                stats['pages_ok'] += 1
            else:
                stats['no_match'] += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1)
                eta = (len(urls) - i - 1) / max(rate, 0.001)
                match_pct = stats['matched_blocks'] / max(stats['total_blocks'], 1) * 100
                print(f'  {i+1:>6}/{len(urls)} pages_ok={stats["pages_ok"]} '
                      f'matched={stats["matched_blocks"]}/{stats["total_blocks"]} ({match_pct:.0f}%) '
                      f'keep={stats["keep"]} discard={stats["discard"]} '
                      f'{rate:.1f}pg/s ETA={eta/60:.0f}m', flush=True)

    elapsed = time.time() - t_start
    match_pct = stats['matched_blocks'] / max(stats['total_blocks'], 1) * 100

    print(f'\nDone in {elapsed:.0f}s', flush=True)
    print(f'  Pages OK: {stats["pages_ok"]}', flush=True)
    print(f'  Simplify fail: {stats["simplify_fail"]}', flush=True)
    print(f'  Features fail: {stats["features_fail"]}', flush=True)
    print(f'  No match: {stats["no_match"]}', flush=True)
    print(f'  Blocks: {stats["matched_blocks"]}/{stats["total_blocks"]} matched ({match_pct:.1f}%)', flush=True)
    print(f'  KEEP: {stats["keep"]} ({stats["keep"]/max(stats["matched_blocks"],1)*100:.1f}%)', flush=True)
    print(f'  DISCARD: {stats["discard"]} ({stats["discard"]/max(stats["matched_blocks"],1)*100:.1f}%)', flush=True)
    print(f'  Saved to {OUTPUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
