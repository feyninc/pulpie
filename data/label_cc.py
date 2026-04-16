"""Label CC pages with DeepSeek V3.2 via AWS Bedrock.

Uses MinerU-HTML's simplification + Dripper v0 prompt.
Saves results incrementally to JSONL (one line per labeled page).
Supports resuming from where it left off.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import boto3

# ── MinerU-HTML module loading (avoids full package import) ──
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

class MinerUHTMLError(Exception):
    pass

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
INPUT_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled.jsonl')
CONCURRENCY = 20
PROMPT_VERSION = 'v0'
MAX_TOKENS = 4096
BEDROCK_REGION = 'us-west-2'
MODEL_ID = 'deepseek.v3.2'


def load_done_urls(output_path):
    """Load URLs already labeled (for resume support)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get('url', ''))
                except json.JSONDecodeError:
                    continue
    return done


def simplify_page(page):
    """Simplify HTML and build prompt. Returns None on failure."""
    try:
        simplified, map_html = simplify_html(page['html'])
        prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
        return {
            'url': page['url'],
            'domain': page['domain'],
            'prompt': prompt,
            'prompt_len': len(prompt),
        }
    except Exception:
        return None


def call_deepseek(client, prompt):
    """Call DeepSeek V3.2 on Bedrock. Returns (response_text, usage) or raises."""
    body = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': MAX_TOKENS,
        'temperature': 0,
    })
    response = client.invoke_model(
        modelId=MODEL_ID, body=body,
        contentType='application/json', accept='application/json',
    )
    result = json.loads(response['body'].read())
    text = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    return text, usage


def process_page(client, prepared, write_lock, out_file, stats):
    """Process a single page: call LLM, parse, write result."""
    t0 = time.time()
    url = prepared['url']

    try:
        response_text, usage = call_deepseek(client, prepared['prompt'])
    except Exception as e:
        stats['llm_fail'] += 1
        return {'url': url, 'status': 'llm_fail', 'error': str(e)[:200]}

    try:
        labels = parse_llm_response(response_text)
    except Exception:
        stats['parse_fail'] += 1
        return {'url': url, 'status': 'parse_fail'}

    n_main = sum(1 for v in labels.values() if v == 'main')
    n_total = len(labels)
    elapsed = time.time() - t0

    result = {
        'url': url,
        'domain': prepared['domain'],
        'status': 'ok',
        'labels': labels,
        'n_main': n_main,
        'n_total': n_total,
        'input_tokens': usage.get('prompt_tokens', 0),
        'output_tokens': usage.get('completion_tokens', 0),
        'latency': round(elapsed, 1),
    }

    # Write immediately (thread-safe)
    with write_lock:
        out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        out_file.flush()
        stats['ok'] += 1
        stats['total_in'] += result['input_tokens']
        stats['total_out'] += result['output_tokens']

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY)
    parser.add_argument('--limit', type=int, default=0, help='Max pages to label (0=all)')
    parser.add_argument('--input', type=str, default=INPUT_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load input pages
    print(f'Loading pages from {args.input}...', flush=True)
    pages = []
    with open(args.input) as f:
        for line in f:
            pages.append(json.loads(line))
    print(f'  Total pages: {len(pages)}', flush=True)

    # Check for already-done pages (resume support)
    done_urls = load_done_urls(args.output)
    if done_urls:
        print(f'  Already labeled: {len(done_urls)} (resuming)', flush=True)
        pages = [p for p in pages if p['url'] not in done_urls]
    print(f'  Pages to label: {len(pages)}', flush=True)

    if args.limit > 0:
        pages = pages[:args.limit]
        print(f'  Limited to: {len(pages)}', flush=True)

    # Step 1: Simplify all pages (CPU-bound, single-threaded)
    print(f'\nSimplifying HTML...', flush=True)
    t0 = time.time()
    prepared = []
    simp_fail = 0
    for i, p in enumerate(pages):
        result = simplify_page(p)
        if result:
            prepared.append(result)
        else:
            simp_fail += 1
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(pages)} simplified, {simp_fail} failed', flush=True)

    simp_time = time.time() - t0
    avg_prompt = sum(p['prompt_len'] for p in prepared) // max(len(prepared), 1)
    print(f'  Done: {len(prepared)}/{len(pages)} in {simp_time:.0f}s ({simp_fail} failed)', flush=True)
    print(f'  Avg prompt: {avg_prompt//1000}K chars', flush=True)

    if not prepared:
        print('No pages to label. Exiting.')
        return

    # Step 2: Label with DeepSeek V3.2
    print(f'\nLabeling with {MODEL_ID} (concurrency={args.concurrency})...', flush=True)
    client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    write_lock = Lock()
    stats = {'ok': 0, 'llm_fail': 0, 'parse_fail': 0, 'total_in': 0, 'total_out': 0}
    t_start = time.time()

    out_file = open(args.output, 'a')

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(process_page, client, p, write_lock, out_file, stats): p
                for p in prepared
            }

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                done_total = stats['ok'] + stats['llm_fail'] + stats['parse_fail']

                if done_total % 50 == 0 or done_total == len(prepared):
                    elapsed = time.time() - t_start
                    rate = done_total / max(elapsed, 1)
                    remaining = (len(prepared) - done_total) / max(rate, 0.001)
                    cost = stats['total_in'] / 1000 * 0.0014 + stats['total_out'] / 1000 * 0.0028
                    print(
                        f'  {done_total:>5}/{len(prepared)} '
                        f'ok={stats["ok"]} llm_fail={stats["llm_fail"]} parse_fail={stats["parse_fail"]} '
                        f'{rate:.2f}pg/s ETA={remaining/60:.0f}m ${cost:.2f}',
                        flush=True,
                    )
    except KeyboardInterrupt:
        print('\n  Interrupted! Results saved so far.', flush=True)
    finally:
        out_file.close()

    # Final summary
    elapsed = time.time() - t_start
    cost = stats['total_in'] / 1000 * 0.0014 + stats['total_out'] / 1000 * 0.0028
    print(f'\n{"="*60}', flush=True)
    print(f'LABELING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  Total time:     {elapsed/3600:.1f}h ({elapsed:.0f}s)', flush=True)
    print(f'  Pages labeled:  {stats["ok"]}', flush=True)
    print(f'  LLM failures:   {stats["llm_fail"]}', flush=True)
    print(f'  Parse failures:  {stats["parse_fail"]}', flush=True)
    print(f'  Throughput:     {stats["ok"]/max(elapsed,1):.2f} pg/s', flush=True)
    print(f'  Tokens in:      {stats["total_in"]:,}', flush=True)
    print(f'  Tokens out:     {stats["total_out"]:,}', flush=True)
    print(f'  Cost:           ${cost:.2f}', flush=True)
    print(f'  Output:         {args.output}', flush=True)

    # Quick stats on labels
    n_empty = 0
    n_total_pages = 0
    with open(args.output) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('status') == 'ok':
                n_total_pages += 1
                if rec.get('n_main', 0) == 0:
                    n_empty += 1
    print(f'  Empty pages:    {n_empty}/{n_total_pages} ({n_empty/max(n_total_pages,1)*100:.1f}%)', flush=True)


if __name__ == '__main__':
    main()
