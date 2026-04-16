"""Analyze whether LLM empty-page labels improve training quality.

Tests whether DeepSeek V3.2's "all blocks are other" judgments are correct,
and whether including such negative examples in training would help the GBM.
"""

import json
import os
import sys
import time

import boto3
import html2text
import jieba
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

# Reuse the MinerU-HTML module patching
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

_pkg = _make_module('mineru_html')
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
for cls_name in ['MinerUHTMLPreprocessError', 'MinerUHTMLPromptError',
                 'MinerUHTMLResponseParseError', 'MinerUHTMLMapToMainError',
                 'MinerUHTMLFallbackError']:
    setattr(_e, cls_name, type(cls_name, (MinerUHTMLError,), {}))
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
_map = _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = _simplify.simplify_html
get_full_prompt = _prompt.get_full_prompt
parse_llm_response = _parse.parse_llm_response
extract_main_html = _map.extract_main_html

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
BENCH_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')


def calc_rouge_n(target, prediction, n=5):
    target, prediction = target.strip(), prediction.strip()
    if not target and not prediction:
        return 1.0
    if not target or not prediction:
        return 0.0
    ref_ng = _create_ngrams(jieba.lcut(target), n)
    pred_ng = _create_ngrams(jieba.lcut(prediction), n)
    return _score_ngrams(ref_ng, pred_ng).fmeasure


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=50)
    args = parser.parse_args()

    client = boto3.client('bedrock-runtime', region_name='us-west-2')

    with open(BENCH_PATH) as f:
        lines = f.readlines()

    results = []
    count = 0
    errors = {'simplify': 0, 'llm': 0, 'parse': 0, 'reconstruct': 0}

    for line in lines:
        rec = json.loads(line)
        if rec.get('meta', {}).get('language') != 'en':
            continue
        html_content = rec.get('html', '')
        reference = rec.get('convert_main_content', '')
        level = rec.get('meta', {}).get('level', '?')
        if not html_content or not reference:
            continue
        if len(html_content) < 500 or len(html_content) > 300000:
            continue

        try:
            simplified, map_html = simplify_html(html_content)
        except Exception:
            errors['simplify'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        prompt = get_full_prompt(simplified, version='v0')
        body = json.dumps({
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 4096,
            'temperature': 0,
        })

        try:
            response = client.invoke_model(
                modelId='deepseek.v3.2', body=body,
                contentType='application/json', accept='application/json',
            )
            result = json.loads(response['body'].read())
            llm_response = result['choices'][0]['message']['content']
        except Exception as e:
            errors['llm'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        try:
            item_labels = parse_llm_response(llm_response)
        except Exception:
            errors['parse'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        n_main = sum(1 for v in item_labels.values() if v == 'main')
        n_total = len(item_labels)
        main_pct = n_main / max(n_total, 1)

        try:
            main_html = extract_main_html(map_html, item_labels)
            h = html2text.HTML2Text(bodywidth=0)
            h.ignore_links = True
            h.ignore_images = True
            pred = h.handle(main_html).strip()
        except Exception:
            errors['reconstruct'] += 1
            pred = ''

        f1 = calc_rouge_n(reference, pred)
        is_empty = (n_main == 0)

        results.append({
            'idx': count, 'level': level, 'f1': f1,
            'n_main': n_main, 'n_total': n_total, 'main_pct': main_pct,
            'ref_len': len(reference), 'pred_len': len(pred), 'is_empty': is_empty,
        })

        tag = 'EMPTY' if is_empty else f'{n_main:>3}/{n_total}'
        print(f'  [{count:>2}] F1={f1:.4f}  {tag:<10}  main%={main_pct:.0%}  ref={len(reference):>5}  pred={len(pred):>5}  {level}', flush=True)

        count += 1
        if count >= args.pages:
            break

    # === ANALYSIS ===
    empty_pages = [r for r in results if r['is_empty']]
    nonempty = [r for r in results if not r['is_empty']]
    low_main = [r for r in results if 0 < r['main_pct'] < 0.1]

    print(f'\n{"="*60}')
    print(f'EMPTY PAGE ANALYSIS ({len(results)} pages)')
    print(f'{"="*60}')
    print(f'  Empty (all other):  {len(empty_pages)} ({len(empty_pages)/max(len(results),1)*100:.1f}%)')
    print(f'  Low main (<10%):    {len(low_main)}')
    print(f'  Non-empty:          {len(nonempty)}')
    print(f'  Errors: {errors}')

    if empty_pages:
        print(f'\n  Empty pages detail:')
        for r in empty_pages:
            # Empty pred but ref exists = LLM thinks no main content
            # If ref is long, this is a mistake. If ref is short, it's correct.
            verdict = 'CORRECT (short ref)' if r['ref_len'] < 200 else 'WRONG (ref has content)'
            print(f'    idx={r["idx"]} level={r["level"]} ref_len={r["ref_len"]:>5} {verdict}')
        avg_ref = sum(r['ref_len'] for r in empty_pages) / len(empty_pages)
        print(f'  Avg reference length on empty pages: {avg_ref:.0f} chars')

    if nonempty:
        avg_f1 = sum(r['f1'] for r in nonempty) / len(nonempty)
        avg_main = sum(r['main_pct'] for r in nonempty) / len(nonempty)
        print(f'\n  Non-empty pages: avg F1={avg_f1:.4f}, avg main%={avg_main:.1%}')

    # Main% buckets
    print(f'\n  Main% vs F1:')
    print(f'  {"Main%":>10} {"Count":>6} {"Avg F1":>8} {"Avg ref":>8} {"Avg pred":>8}')
    print(f'  {"-"*44}')
    for lo, hi, label in [(0, 0.001, '0% (empty)'), (0.001, 0.05, '<5%'), (0.05, 0.15, '5-15%'),
                           (0.15, 0.3, '15-30%'), (0.3, 0.5, '30-50%'), (0.5, 1.01, '50-100%')]:
        sub = [r for r in results if lo <= r['main_pct'] < hi]
        if sub:
            avg_f = sum(r['f1'] for r in sub) / len(sub)
            avg_ref = sum(r['ref_len'] for r in sub) / len(sub)
            avg_pred = sum(r['pred_len'] for r in sub) / len(sub)
            print(f'  {label:>10} {len(sub):>6} {avg_f:>8.4f} {avg_ref:>8.0f} {avg_pred:>8.0f}')

    # Training value analysis
    print(f'\n  TRAINING VALUE OF EMPTY PAGES:')
    print(f'  If LLM labels all blocks as "other" on a page:')
    print(f'    - Every block becomes a NEGATIVE example (label=0)')
    print(f'    - This teaches GBM to discard boilerplate-only pages')
    print(f'    - Current GBM has no such examples (trained only on pages WITH main content)')
    print(f'    - Even wrong empties (FN) are useful: they represent hard pages where')
    print(f'      the content is genuinely hard to distinguish from boilerplate')

    # Pages where LLM keeps very little but ground truth has content
    precision_traps = [r for r in results if r['main_pct'] < 0.05 and r['ref_len'] > 500]
    if precision_traps:
        print(f'\n  PRECISION TRAP pages (LLM keeps <5% but ref >500 chars): {len(precision_traps)}')
        for r in precision_traps:
            print(f'    idx={r["idx"]} main={r["n_main"]}/{r["n_total"]} ref_len={r["ref_len"]} F1={r["f1"]:.4f} {r["level"]}')


if __name__ == '__main__':
    main()
