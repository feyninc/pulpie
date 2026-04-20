"""Generate teacher logits from EuroBERT-2.1B for distillation.

Runs the teacher model on all ~99K CC pages using multiple GPUs via
torch.multiprocessing. Saves logits as .pt files sharded by page.

Phase 1 of distillation pipeline. Phase 2: train_distill_eurobert.py

Usage:
  python data/generate_teacher_logits.py --num-gpus 4 --batch-size 8
  python data/generate_teacher_logits.py --limit 200 --num-gpus 1  # quick test
"""

import json
import os
import re
import sys
import time
import torch
import torch.multiprocessing as mp
from collections import defaultdict

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

if 'mineru_html' not in sys.modules:
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
    _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
    _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
    _load_file('mineru_html.process.parse_result', 'parse_result.py')
    _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html

from block_chunker import extract_blocks, tokenize_blocks, pack_chunks, SEP_TOKEN
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ── Config ──
TEACHER_PATH = os.path.join(SCRIPT_DIR, 'block_classifier_eurobert_2.1B', 'checkpoint-5250')
MAX_TOKENS = 8192

SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
DOUBLECHECK_PATH = os.path.join(SCRIPT_DIR, 'cc_doublecheck_results.jsonl')
DRIPPER_LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_dripper_83k.jsonl')
DRIPPER_SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled_100k.jsonl')

LOGITS_DIR = os.path.join(SCRIPT_DIR, 'teacher_logits_2.1B')

LABEL_OTHER = 0
LABEL_MAIN = 1
IGNORE_LABEL = -100


def load_cc_pages(min_agreement=0.70, limit=0):
    labels_by_url = {}

    if os.path.exists(DOUBLECHECK_PATH) and os.path.exists(LABELED_PATH):
        agreement_by_url = {}
        with open(DOUBLECHECK_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'ok':
                    agreement_by_url[r['url']] = r['agreement_rate']

        with open(LABELED_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') != 'ok':
                    continue
                url = r['url']
                agr = agreement_by_url.get(url)
                if agr is None or agr < min_agreement:
                    continue
                labels_by_url[url] = r.get('labels', {})

        print(f'  Original labeled (agreement >= {min_agreement}): {len(labels_by_url)}', flush=True)

    dripper_count = 0
    if os.path.exists(DRIPPER_LABELED_PATH):
        with open(DRIPPER_LABELED_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') != 'ok':
                    continue
                url = r['url']
                if url not in labels_by_url:
                    labels_by_url[url] = r.get('labels', {})
                    dripper_count += 1
        print(f'  Dripper-only labeled (new): {dripper_count}', flush=True)

    print(f'  Total unique labeled URLs: {len(labels_by_url)}', flush=True)

    sampled_paths = [SAMPLED_PATH, DRIPPER_SAMPLED_PATH]
    pages = []
    for sampled_path in sampled_paths:
        if not os.path.exists(sampled_path):
            continue
        with open(sampled_path) as f:
            for line in f:
                r = json.loads(line)
                if r['url'] in labels_by_url:
                    pages.append({
                        'url': r['url'],
                        'html': r['html'],
                        'labels': labels_by_url.pop(r['url']),
                    })

    print(f'  Pages with HTML + labels: {len(pages)}', flush=True)
    if limit > 0:
        pages = pages[:limit]
    return pages


def build_chunk_examples(simplified_html, labels_dict, tokenizer, sep_token_id):
    blocks = extract_blocks(simplified_html)
    if not blocks:
        return []

    item_id_pattern = re.compile(r'_item_id="(\d+)"')
    block_item_ids = []
    for block in blocks:
        m = item_id_pattern.search(block)
        block_item_ids.append(m.group(1) if m else None)

    block_labels = []
    valid_blocks = []
    for i, (block, item_id) in enumerate(zip(blocks, block_item_ids)):
        if item_id is None:
            continue
        label_str = labels_dict.get(item_id)
        if label_str is None:
            continue
        valid_blocks.append(block)
        block_labels.append(LABEL_MAIN if label_str == 'main' else LABEL_OTHER)

    if not valid_blocks:
        return []

    block_token_ids = tokenize_blocks(valid_blocks, tokenizer)
    chunks = pack_chunks(
        block_token_ids, max_tokens=MAX_TOKENS,
        sep_token_id=sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    examples = []
    for chunk_ids, chunk_block_indices in chunks:
        token_labels = [IGNORE_LABEL] * len(chunk_ids)
        sep_count = 0
        for tok_idx, tid in enumerate(chunk_ids):
            if tid == sep_token_id:
                if sep_count < len(chunk_block_indices):
                    label_idx = chunk_block_indices[sep_count]
                    token_labels[tok_idx] = block_labels[label_idx]
                sep_count += 1

        n_labeled = sum(1 for l in token_labels if l != IGNORE_LABEL)
        if n_labeled == 0:
            continue

        examples.append({
            'input_ids': chunk_ids,
            'labels': token_labels,
            'n_blocks': n_labeled,
            'n_main': sum(1 for l in token_labels if l == LABEL_MAIN),
        })

    return examples


def worker_fn(gpu_id, chunk_indices, all_input_ids, output_dir, batch_size):
    """Worker that runs teacher inference on a shard of chunks."""
    device = f'cuda:{gpu_id}'
    print(f'  [GPU {gpu_id}] Loading teacher model...', flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        TEACHER_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()

    n = len(chunk_indices)
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_idx = chunk_indices[batch_start:batch_start + batch_size]
        batch_input_ids = [all_input_ids[i] for i in batch_idx]
        max_len = max(len(ids) for ids in batch_input_ids)

        padded_ids = []
        attn_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attn_mask_t = torch.tensor(attn_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits

        for j, global_idx in enumerate(batch_idx):
            seq_len = len(all_input_ids[global_idx])
            chunk_logits = logits[j, :seq_len, :].cpu().to(torch.float16)
            torch.save(chunk_logits, os.path.join(output_dir, f'{global_idx}.pt'))

        done = min(batch_start + batch_size, n)
        if done % (batch_size * 50) == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f'  [GPU {gpu_id}] {done}/{n} ({elapsed:.0f}s, {rate:.1f} chunks/s)', flush=True)

    print(f'  [GPU {gpu_id}] Done in {time.time() - t0:.0f}s', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    print('=== Phase 1: Generate Teacher Logits ===', flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [SEP_TOKEN]})
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f'  <|sep|> id = {sep_token_id}', flush=True)

    # Load pages and build chunks
    print('\nLoading CC pages...', flush=True)
    pages = load_cc_pages(limit=args.limit)
    print(f'  {len(pages)} pages loaded', flush=True)

    print('\nBuilding chunks...', flush=True)
    all_examples = []
    fail = 0
    for i, page in enumerate(pages):
        try:
            simplified, _ = simplify_html(page['html'])
        except Exception:
            fail += 1
            continue
        chunk_examples = build_chunk_examples(simplified, page['labels'], tokenizer, sep_token_id)
        if not chunk_examples:
            fail += 1
            continue
        all_examples.extend(chunk_examples)
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{len(pages)}: {len(all_examples)} chunks, {fail} failed', flush=True)

    print(f'  Done: {len(all_examples)} chunks, {fail} failed', flush=True)

    # Save chunk metadata (input_ids + labels) to disk
    os.makedirs(LOGITS_DIR, exist_ok=True)
    meta_path = os.path.join(LOGITS_DIR, 'chunks_meta.pt')
    print(f'\nSaving chunk metadata to {meta_path}...', flush=True)
    torch.save({
        'input_ids': [e['input_ids'] for e in all_examples],
        'labels': [e['labels'] for e in all_examples],
        'n_blocks': [e['n_blocks'] for e in all_examples],
        'n_main': [e['n_main'] for e in all_examples],
    }, meta_path)

    all_input_ids = [e['input_ids'] for e in all_examples]
    n_chunks = len(all_input_ids)
    print(f'  {n_chunks} chunks saved', flush=True)

    # Shard across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f'\nRunning teacher inference on {num_gpus} GPUs, batch_size={args.batch_size}...', flush=True)

    indices = list(range(n_chunks))
    shard_size = (n_chunks + num_gpus - 1) // num_gpus
    shards = [indices[i*shard_size:(i+1)*shard_size] for i in range(num_gpus)]

    t0 = time.time()
    if num_gpus == 1:
        worker_fn(0, shards[0], all_input_ids, LOGITS_DIR, args.batch_size)
    else:
        mp.set_start_method('spawn', force=True)
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_fn,
                args=(gpu_id, shards[gpu_id], all_input_ids, LOGITS_DIR, args.batch_size),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    total_time = time.time() - t0
    print(f'\nTeacher inference done in {total_time:.0f}s ({n_chunks/total_time:.1f} chunks/s)', flush=True)

    # Verify all logits saved
    saved = len([f for f in os.listdir(LOGITS_DIR) if f.endswith('.pt') and f != 'chunks_meta.pt'])
    print(f'Saved {saved}/{n_chunks} logit files to {LOGITS_DIR}/', flush=True)


if __name__ == '__main__':
    main()
