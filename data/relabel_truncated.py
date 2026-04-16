"""Re-label truncated pages with higher MAX_TOKENS.

Reads cc_relabel_urls.json (43 URLs), re-runs DeepSeek V3.2 with
MAX_TOKENS=16384, and patches cc_labeled.jsonl in place.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import boto3

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
MAX_TOKENS = 16384
BEDROCK_REGION = 'us-west-2'
MODEL_ID = 'deepseek.v3.2'
CONCURRENCY = 10
PROMPT_VERSION = 'v0'

SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled.jsonl')
RELABEL_URLS_PATH = os.path.join(SCRIPT_DIR, 'cc_relabel_urls.json')


def main():
    with open(RELABEL_URLS_PATH) as f:
        relabel_urls = set(json.load(f))
    print(f'URLs to re-label: {len(relabel_urls)}', flush=True)

    # Load HTML for these URLs
    print('Loading HTML from cc_sampled.jsonl...', flush=True)
    pages = {}
    with open(SAMPLED_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r['url'] in relabel_urls:
                pages[r['url']] = r
    print(f'  Found HTML for {len(pages)}/{len(relabel_urls)} pages', flush=True)

    # Simplify
    print('Simplifying HTML...', flush=True)
    prepared = []
    for url, page in pages.items():
        try:
            simplified, _ = simplify_html(page['html'])
            prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
            prepared.append({
                'url': url,
                'domain': page['domain'],
                'prompt': prompt,
            })
        except Exception:
            print(f'  SKIP simplify fail: {page["domain"]}', flush=True)
    print(f'  Prepared: {len(prepared)}', flush=True)

    # Label with higher MAX_TOKENS
    print(f'\nLabeling with {MODEL_ID} (MAX_TOKENS={MAX_TOKENS}, concurrency={CONCURRENCY})...', flush=True)
    client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    results = {}
    stats = {'ok': 0, 'fail': 0}
    t_start = time.time()

    def process_one(p):
        t0 = time.time()
        body = json.dumps({
            'messages': [{'role': 'user', 'content': p['prompt']}],
            'max_tokens': MAX_TOKENS,
            'temperature': 0,
        })
        try:
            response = client.invoke_model(
                modelId=MODEL_ID, body=body,
                contentType='application/json', accept='application/json',
            )
            result = json.loads(response['body'].read())
            text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
        except Exception as e:
            return {'url': p['url'], 'status': 'llm_fail', 'error': str(e)[:200]}

        try:
            labels = parse_llm_response(text)
        except Exception:
            return {'url': p['url'], 'status': 'parse_fail'}

        n_main = sum(1 for v in labels.values() if v == 'main')
        return {
            'url': p['url'],
            'domain': p['domain'],
            'status': 'ok',
            'labels': labels,
            'n_main': n_main,
            'n_total': len(labels),
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'latency': round(time.time() - t0, 1),
        }

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(process_one, p): p for p in prepared}
        for future in as_completed(futures):
            r = future.result()
            url = r['url']
            domain = r.get('domain', '?')
            if r['status'] == 'ok':
                results[url] = r
                stats['ok'] += 1
                print(f'  OK  {domain:<40} {r["n_main"]:>4}/{r["n_total"]} main  out={r["output_tokens"]:>5}  {r["latency"]:.0f}s', flush=True)
            else:
                stats['fail'] += 1
                print(f'  FAIL {domain:<40} {r["status"]}', flush=True)

    elapsed = time.time() - t_start
    print(f'\nDone: {stats["ok"]} ok, {stats["fail"]} failed in {elapsed:.0f}s', flush=True)

    # Patch cc_labeled.jsonl — replace old entries with new ones
    print(f'\nPatching {LABELED_PATH}...', flush=True)
    patched = 0
    kept = 0
    lines_out = []
    with open(LABELED_PATH) as f:
        for line in f:
            r = json.loads(line)
            url = r.get('url', '')
            if url in results:
                lines_out.append(json.dumps(results[url], ensure_ascii=False) + '\n')
                patched += 1
            else:
                lines_out.append(line)
                kept += 1

    with open(LABELED_PATH, 'w') as f:
        f.writelines(lines_out)

    print(f'  Patched: {patched}, Kept: {kept}', flush=True)

    # Verify improvement
    print(f'\nVerification — re-labeled pages:', flush=True)
    for url, r in sorted(results.items(), key=lambda x: -x[1].get('output_tokens', 0)):
        still_trunc = '⚠ STILL TRUNCATED' if r.get('output_tokens', 0) >= 16000 else ''
        print(f'  {r["domain"]:<40} {r["n_main"]:>4}/{r["n_total"]} main  out={r["output_tokens"]:>5} {still_trunc}', flush=True)


if __name__ == '__main__':
    main()
