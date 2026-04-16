"""Double-check DeepSeek V3.2 labels using Dripper 0.6B model.

Runs the Dripper model (via local vLLM) on the same CC pages and
compares its labels against DeepSeek's. Reports agreement rates
and flags disagreements.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
_prompt = _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
_parse = _load_file('mineru_html.process.parse_result', 'parse_result.py')

simplify_html = _simplify.simplify_html
get_full_prompt = _prompt.get_full_prompt
parse_llm_response = _parse.parse_llm_response

# ── Config ──
VLLM_URL = "http://localhost:8235/v1/chat/completions"
MODEL_NAME = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
PROMPT_VERSION = 'short_compact'
MAX_TOKENS = 4096
CONCURRENCY = 32

SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'cc_doublecheck_results.jsonl')


def extract_item_ids(html_str):
    """Extract _item_id values from simplified HTML."""
    return [int(m) for m in re.findall(r'_item_id="(\d+)"', html_str)]


def build_guided_regex(item_ids):
    """Build guided_regex pattern matching Dripper's compact format."""
    item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
    return f'<answer>\\s*{item_pattern}\\s*</answer>'


def call_dripper(prompt, item_ids=None):
    """Call Dripper model via vLLM OpenAI API with guided decoding."""
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    if item_ids:
        body["guided_regex"] = build_guided_regex(item_ids)
    resp = requests.post(VLLM_URL, json=body, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    text = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    return text, usage


def process_page(page, deepseek_labels):
    """Simplify page, run Dripper, compare with DeepSeek labels."""
    url = page['url']
    domain = page['domain']

    try:
        simplified, _ = simplify_html(page['html'])
        prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
    except Exception:
        return {'url': url, 'domain': domain, 'status': 'simplify_fail'}

    try:
        response_text, usage = call_dripper(prompt)
    except Exception as e:
        return {'url': url, 'domain': domain, 'status': 'llm_fail', 'error': str(e)[:200]}

    try:
        dripper_labels = parse_llm_response(response_text)
    except Exception:
        return {'url': url, 'domain': domain, 'status': 'parse_fail'}

    # Compare with DeepSeek labels
    ds_labels = deepseek_labels.get(url, {})
    all_ids = set(dripper_labels.keys()) | set(ds_labels.keys())
    common_ids = set(dripper_labels.keys()) & set(ds_labels.keys())

    agree = 0
    disagree_ds_main = 0  # DeepSeek says main, Dripper says other
    disagree_dr_main = 0  # Dripper says main, DeepSeek says other
    for bid in common_ids:
        if dripper_labels[bid] == ds_labels[bid]:
            agree += 1
        elif ds_labels[bid] == 'main':
            disagree_ds_main += 1
        else:
            disagree_dr_main += 1

    dripper_only = len(set(dripper_labels.keys()) - set(ds_labels.keys()))
    ds_only = len(set(ds_labels.keys()) - set(dripper_labels.keys()))

    return {
        'url': url,
        'domain': domain,
        'status': 'ok',
        'dripper_n_total': len(dripper_labels),
        'dripper_n_main': sum(1 for v in dripper_labels.values() if v == 'main'),
        'ds_n_total': len(ds_labels),
        'ds_n_main': sum(1 for v in ds_labels.values() if v == 'main'),
        'common_blocks': len(common_ids),
        'agree': agree,
        'disagree_ds_main': disagree_ds_main,
        'disagree_dr_main': disagree_dr_main,
        'dripper_only': dripper_only,
        'ds_only': ds_only,
        'agreement_rate': agree / max(len(common_ids), 1),
        'input_tokens': usage.get('prompt_tokens', 0),
        'output_tokens': usage.get('completion_tokens', 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY)
    args = parser.parse_args()

    # Load DeepSeek labels
    print('Loading DeepSeek labels...', flush=True)
    deepseek_labels = {}
    deepseek_urls = set()
    with open(LABELED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok':
                deepseek_labels[r['url']] = r.get('labels', {})
                deepseek_urls.add(r['url'])
    print(f'  {len(deepseek_labels)} labeled pages', flush=True)

    # Load HTML
    print('Loading HTML from cc_sampled.jsonl...', flush=True)
    pages = []
    with open(SAMPLED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r['url'] in deepseek_urls:
                pages.append(r)
    print(f'  Found {len(pages)} pages with HTML', flush=True)

    if args.limit > 0:
        pages = pages[:args.limit]
        print(f'  Limited to {len(pages)}', flush=True)

    # Simplify all pages first (CPU-bound)
    print('\nSimplifying HTML...', flush=True)
    prepared = []
    simp_fail = 0
    for i, page in enumerate(pages):
        try:
            simplified, _ = simplify_html(page['html'])
            prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
            item_ids = extract_item_ids(simplified)
            prepared.append({
                'url': page['url'],
                'domain': page['domain'],
                'prompt': prompt,
                'item_ids': item_ids,
            })
        except Exception:
            simp_fail += 1
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(pages)} simplified, {simp_fail} failed', flush=True)
    print(f'  Done: {len(prepared)} prepared, {simp_fail} failed', flush=True)

    # Run Dripper on all pages
    print(f'\nRunning Dripper (concurrency={args.concurrency})...', flush=True)
    write_lock = Lock()
    stats = {'ok': 0, 'llm_fail': 0, 'parse_fail': 0}
    t_start = time.time()

    out_file = open(OUTPUT_PATH, 'w')

    def do_one(p):
        url = p['url']
        domain = p['domain']
        try:
            response_text, usage = call_dripper(p['prompt'], item_ids=p.get('item_ids'))
        except Exception as e:
            return {'url': url, 'domain': domain, 'status': 'llm_fail', 'error': str(e)[:200]}

        try:
            dripper_labels = parse_llm_response(response_text)
        except Exception:
            return {'url': url, 'domain': domain, 'status': 'parse_fail'}

        ds_labels = deepseek_labels.get(url, {})
        common_ids = set(dripper_labels.keys()) & set(ds_labels.keys())

        agree = disagree_ds_main = disagree_dr_main = 0
        for bid in common_ids:
            if dripper_labels[bid] == ds_labels[bid]:
                agree += 1
            elif ds_labels[bid] == 'main':
                disagree_ds_main += 1
            else:
                disagree_dr_main += 1

        return {
            'url': url, 'domain': domain, 'status': 'ok',
            'dripper_n_main': sum(1 for v in dripper_labels.values() if v == 'main'),
            'dripper_n_total': len(dripper_labels),
            'ds_n_main': sum(1 for v in ds_labels.values() if v == 'main'),
            'ds_n_total': len(ds_labels),
            'common': len(common_ids),
            'agree': agree,
            'disagree_ds_main': disagree_ds_main,
            'disagree_dr_main': disagree_dr_main,
            'agreement_rate': agree / max(len(common_ids), 1),
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
        }

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(do_one, p): p for p in prepared}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                with write_lock:
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_file.flush()
                    if result['status'] == 'ok':
                        stats['ok'] += 1
                    else:
                        stats[result['status']] = stats.get(result['status'], 0) + 1

                done = sum(stats.values())
                if done % 200 == 0 or done == len(prepared):
                    elapsed = time.time() - t_start
                    rate = done / max(elapsed, 1)
                    eta = (len(prepared) - done) / max(rate, 0.001)
                    print(f'  {done:>5}/{len(prepared)} ok={stats["ok"]} fail={stats.get("llm_fail",0)+stats.get("parse_fail",0)} '
                          f'{rate:.1f}pg/s ETA={eta/60:.0f}m', flush=True)
    except KeyboardInterrupt:
        print('\n  Interrupted!', flush=True)
    finally:
        out_file.close()

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed:.0f}s ({stats["ok"]}/{len(prepared)} ok)', flush=True)

    # Analyze results
    print(f'\n{"="*60}')
    print(f'AGREEMENT ANALYSIS')
    print(f'{"="*60}')

    results = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'ok':
                results.append(r)

    total_common = sum(r['common'] for r in results)
    total_agree = sum(r['agree'] for r in results)
    total_ds_main = sum(r['disagree_ds_main'] for r in results)
    total_dr_main = sum(r['disagree_dr_main'] for r in results)

    print(f'  Pages analyzed: {len(results)}')
    print(f'  Common blocks: {total_common:,}')
    print(f'  Agreement: {total_agree:,} ({total_agree/max(total_common,1)*100:.1f}%)')
    print(f'  DeepSeek=main, Dripper=other: {total_ds_main:,} ({total_ds_main/max(total_common,1)*100:.1f}%)')
    print(f'  Dripper=main, DeepSeek=other: {total_dr_main:,} ({total_dr_main/max(total_common,1)*100:.1f}%)')

    # Per-page agreement distribution
    import numpy as np
    rates = [r['agreement_rate'] for r in results]
    print(f'\n  Per-page agreement rate:')
    print(f'    Mean:   {np.mean(rates):.3f}')
    print(f'    Median: {np.median(rates):.3f}')
    print(f'    P5:     {np.percentile(rates, 5):.3f}')
    print(f'    P25:    {np.percentile(rates, 25):.3f}')
    print(f'    P75:    {np.percentile(rates, 75):.3f}')
    print(f'    P95:    {np.percentile(rates, 95):.3f}')

    # Pages with low agreement
    low_agree = [r for r in results if r['agreement_rate'] < 0.7]
    print(f'\n  Pages with <70% agreement: {len(low_agree)} ({len(low_agree)/len(results)*100:.1f}%)')
    for r in sorted(low_agree, key=lambda x: x['agreement_rate'])[:10]:
        print(f'    {r["domain"]:<40} agree={r["agreement_rate"]:.1%}  ds={r["ds_n_main"]}/{r["ds_n_total"]}  dr={r["dripper_n_main"]}/{r["dripper_n_total"]}')


if __name__ == '__main__':
    main()
