"""Test DeepSeek V3.2 + MinerU-HTML simplification pipeline on WebMainBench.

Validates that we can reproduce ~0.91 ROUGE-5 using:
1. MinerU-HTML's simplify_html for block segmentation
2. Dripper's v0 prompt for classification
3. DeepSeek V3.2 on AWS Bedrock for inference
4. MinerU-HTML's extract_main_html for reconstruction
5. html2text for canonicalization
6. jieba-based ROUGE-5 for scoring (matching paper)
"""

import json
import os
import sys
import time

import boto3
import html2text
import jieba
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

# Add MinerU-HTML to path - import only the process modules we need
# We create fake parent modules to avoid importing transformers/vllm
MINERU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'MinerU-HTML')

import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# Create stub modules that the process files depend on
def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod

# mineru_html package stub (blocks __init__.py from running)
_pkg = _make_module('mineru_html')

# constants
_c = _make_module('mineru_html.constants')
_c.ITEM_ID_ATTR = '_item_id'
_c.TAIL_BLOCK_TAG = 'cc-alg-uc-text'
_c.SELECT_ATTR = 'cc-select'
_c.CLASS_ATTR = 'mark-selected'
class TagType(Enum):
    Main = 'main'
    Other = 'other'
_c.TagType = TagType

# exceptions
_e = _make_module('mineru_html.exceptions')
class MinerUHTMLError(Exception): pass
for cls_name in ['MinerUHTMLPreprocessError', 'MinerUHTMLPromptError',
                 'MinerUHTMLResponseParseError', 'MinerUHTMLMapToMainError',
                 'MinerUHTMLFallbackError']:
    cls = type(cls_name, (MinerUHTMLError,), {})
    setattr(_e, cls_name, cls)
_e.MinerUHTMLError = MinerUHTMLError

# base (dataclasses)
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

# process package stub
_make_module('mineru_html.process')

# Now load the actual process modules from files
def _load_file(mod_name, filename):
    path = os.path.join(MINERU_PATH, 'mineru_html', 'process', filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_html_utils = _load_file('mineru_html.process.html_utils', 'html_utils.py')
_simplify = _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
_prompt = _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
_parse = _load_file('mineru_html.process.parse_result', 'parse_result.py')
_map = _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = _simplify.simplify_html
get_full_prompt = _prompt.get_full_prompt
parse_llm_response = _parse.parse_llm_response
extract_main_html = _map.extract_main_html


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BENCH_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')


def calc_rouge_n(target, prediction, n=5):
    """ROUGE-N with jieba tokenization (matching paper)."""
    target = target.strip()
    prediction = prediction.strip()
    if not target and not prediction:
        return {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}
    ref_tokens = jieba.lcut(target)
    pred_tokens = jieba.lcut(prediction)
    ref_ng = _create_ngrams(ref_tokens, n)
    pred_ng = _create_ngrams(pred_tokens, n)
    score = _score_ngrams(ref_ng, pred_ng)
    return {'prec': score.precision, 'rec': score.recall, 'f1': score.fmeasure}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=20)
    parser.add_argument('--prompt-version', type=str, default='v0')
    args = parser.parse_args()

    client = boto3.client('bedrock-runtime', region_name='us-west-2')

    with open(BENCH_PATH) as f:
        lines = f.readlines()

    scores = []
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
        if len(html_content) < 500:
            continue

        # Step 1: Simplify
        try:
            simplified, map_html = simplify_html(html_content)
        except Exception:
            errors['simplify'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        # Step 2: Build prompt
        prompt = get_full_prompt(simplified, version=args.prompt_version)

        # Step 3: Call DeepSeek V3.2
        body = json.dumps({
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 4096,
            'temperature': 0,
        })

        t0 = time.time()
        try:
            response = client.invoke_model(
                modelId='deepseek.v3.2',
                body=body,
                contentType='application/json',
                accept='application/json',
            )
            result = json.loads(response['body'].read())
            llm_response = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            elapsed = time.time() - t0
        except Exception as e:
            errors['llm'] += 1
            print(f'  [{count:>2}] LLM error: {e}')
            count += 1
            if count >= args.pages:
                break
            continue

        # Step 4: Parse
        try:
            item_labels = parse_llm_response(llm_response)
        except Exception:
            errors['parse'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        # Step 5: Reconstruct
        try:
            main_html = extract_main_html(map_html, item_labels)
        except Exception:
            errors['reconstruct'] += 1
            count += 1
            if count >= args.pages:
                break
            continue

        # Step 6: html2text
        h = html2text.HTML2Text(bodywidth=0)
        h.ignore_links = True
        h.ignore_images = True
        pred = h.handle(main_html).strip()

        # Step 7: Score
        score = calc_rouge_n(reference, pred)
        scores.append({**score, 'level': level})

        n_main = sum(1 for v in item_labels.values() if v == 'main')
        n_total = len(item_labels)
        print(f'  [{count:>2}] F1={score["f1"]:.4f} P={score["prec"]:.4f} R={score["rec"]:.4f}  {level:<7} {n_main}/{n_total} blocks  in={usage.get("prompt_tokens",0):>5} out={usage.get("completion_tokens",0):>4}  {elapsed:.1f}s')

        count += 1
        if count >= args.pages:
            break

    print(f'\nErrors: {errors}')
    if scores:
        avg_f1 = sum(s['f1'] for s in scores) / len(scores)
        avg_p = sum(s['prec'] for s in scores) / len(scores)
        avg_r = sum(s['rec'] for s in scores) / len(scores)
        print(f'\nDeepSeek V3.2 + MinerU-HTML pipeline ({len(scores)}/{count} pages):')
        print(f'  Jieba ROUGE-5: P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}')
        print(f'  Paper reports:                         F1=0.9098')

        for lev in ['simple', 'mid', 'hard']:
            sub = [s for s in scores if s['level'] == lev]
            if sub:
                print(f'  {lev:>7}: F1={sum(s["f1"] for s in sub)/len(sub):.4f} ({len(sub)} pages)')


if __name__ == '__main__':
    main()
