"""Train [BLOCK] marker classifier on Qwen3-Embedding-0.6B for block-level extraction.

PL-Marker approach:
- Insert a [BLOCK] token before each _item_id block in simplified HTML
- Encoder processes the full page with bidirectional attention (SDPA, is_causal=False)
- Binary classification (main=1, other=0) at [BLOCK] positions only
- All other token positions get IGNORE_LABEL (-100)

Usage:
  # Single GPU quick test
  python data/train_bio_classifier.py --limit 200 --epochs 1

  # Full multi-GPU training (launched via accelerate)
  accelerate launch --num_processes 4 data/train_bio_classifier.py --epochs 3
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import html2text
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from sklearn.metrics import f1_score, precision_score, recall_score

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
_map_main = _load_file('mineru_html.process.map_to_main', 'map_to_main.py')
simplify_html = _simplify.simplify_html
extract_main_html = _map_main.extract_main_html

# ── Config ──
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BLOCK_TOKEN = "[BLOCK]"
MAX_LENGTH = 32768
SEED = 42

SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
DOUBLECHECK_PATH = os.path.join(SCRIPT_DIR, 'cc_doublecheck_results.jsonl')
WMB_EVAL_PATH = os.path.join(SCRIPT_DIR, 'wmb_eval_sample.jsonl')
# Dripper-only labeled data (83K pages from scaled CC)
DRIPPER_LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_dripper_83k.jsonl')
DRIPPER_SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled_100k.jsonl')
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'block_classifier')

# Binary labels at [BLOCK] positions
LABEL_OTHER = 0
LABEL_MAIN = 1
IGNORE_LABEL = -100
NUM_LABELS = 2


def make_model_bidirectional(model):
    """Disable causal masking on all attention layers for bidirectional SDPA.

    When is_causal=False and attention_mask=None, SDPA runs fully
    bidirectional using its efficient flash/mem-efficient kernels.
    """
    for module in model.modules():
        if hasattr(module, 'is_causal'):
            module.is_causal = False


def insert_block_markers(simplified_html, labels_dict):
    """Insert [BLOCK] tokens before each _item_id block and build labels.

    Returns:
        (marked_html, block_labels) where block_labels is a list of 0/1
        in order of [BLOCK] token appearance, or None if no blocks found.
    """
    pattern = re.compile(r'(_item_id="(\d+)")')
    block_labels = []
    parts = []
    last_end = 0

    for m in pattern.finditer(simplified_html):
        item_id = m.group(2)
        label_str = labels_dict.get(item_id)
        if label_str is None:
            continue

        is_main = LABEL_MAIN if label_str == 'main' else LABEL_OTHER

        parts.append(simplified_html[last_end:m.start()])
        parts.append(BLOCK_TOKEN + ' ')
        parts.append(m.group(0))
        last_end = m.end()
        block_labels.append(is_main)

    if not block_labels:
        return None

    parts.append(simplified_html[last_end:])
    marked_html = ''.join(parts)
    return marked_html, block_labels


def build_block_example(simplified_html, labels_dict, tokenizer, block_token_id):
    """Build a training example with [BLOCK] marker tokens and labels."""
    result = insert_block_markers(simplified_html, labels_dict)
    if result is None:
        return None
    marked_html, block_labels = result

    encoding = tokenizer(
        marked_html,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding=False,
    )

    input_ids = encoding['input_ids']

    # IGNORE_LABEL everywhere, actual label at [BLOCK] positions
    token_labels = [IGNORE_LABEL] * len(input_ids)
    block_idx = 0
    for tok_idx, tid in enumerate(input_ids):
        if tid == block_token_id:
            if block_idx < len(block_labels):
                token_labels[tok_idx] = block_labels[block_idx]
                block_idx += 1

    n_labeled = sum(1 for l in token_labels if l != IGNORE_LABEL)
    if n_labeled == 0:
        return None

    return {
        'input_ids': input_ids,
        'labels': token_labels,
        'n_blocks': n_labeled,
        'n_main': sum(1 for l in token_labels if l == LABEL_MAIN),
    }


class BlockDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        return {'input_ids': e['input_ids'], 'labels': e['labels']}


class BlockCollator:
    """Collator with dict-wrapped attention mask for bidirectional SDPA.

    Passes attention_mask as {"full_attention": None} to bypass Qwen3's
    causal mask creation entirely. Combined with is_causal=False on attention
    modules, SDPA runs fully bidirectional using efficient kernels.

    Padding tokens are masked only via labels (IGNORE_LABEL) — the model
    attends to pad tokens but only classifies at [BLOCK] positions, so
    pad attention is wasted compute but doesn't affect correctness.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)

        input_ids = []
        labels = []

        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.pad_token_id] * pad_len)
            labels.append(f['labels'] + [IGNORE_LABEL] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            # Dict bypasses create_causal_mask; None = no mask = bidirectional
            'attention_mask': {'full_attention': None},
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class BidirectionalTrainer(Trainer):
    """Trainer that wraps attention_mask in a dict for bidirectional SDPA
    and applies class-weighted CE loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        # attention_mask is already a dict from BlockCollator
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weight,
            ignore_index=IGNORE_LABEL,
        )
        return (loss, outputs) if return_outputs else loss


def load_cc_pages(min_agreement=0.70, limit=0, original_only=False):
    """Load CC pages from original (DeepSeek+Dripper) and optionally new Dripper-only data."""
    labels_by_url = {}

    # 1) Original DeepSeek-labeled data (with agreement filtering)
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

    # 2) Dripper-only labeled data (83K scaled pages, no agreement filter needed)
    dripper_count = 0
    if not original_only and os.path.exists(DRIPPER_LABELED_PATH):
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
    elif original_only:
        print(f'  Skipping Dripper-only data (--original-only)', flush=True)

    print(f'  Total unique labeled URLs: {len(labels_by_url)}', flush=True)

    # Load HTML from sampled files
    sampled_paths = [SAMPLED_PATH]
    if not original_only:
        sampled_paths.append(DRIPPER_SAMPLED_PATH)
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


def prepare_examples(tokenizer, block_token_id, limit=0, max_length=MAX_LENGTH, original_only=False):
    """Prepare block-marker-labeled examples from CC data."""
    print('Loading CC pages...', flush=True)
    pages = load_cc_pages(limit=limit, original_only=original_only)
    print(f'  {len(pages)} pages loaded', flush=True)

    examples = []
    fail = 0
    too_long = 0

    for i, page in enumerate(pages):
        try:
            simplified, _ = simplify_html(page['html'])
        except Exception:
            fail += 1
            continue

        result = build_block_example(simplified, page['labels'], tokenizer, block_token_id)
        if result is None:
            fail += 1
            continue

        if len(result['input_ids']) > max_length:
            too_long += 1
            continue

        examples.append(result)

        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(pages)}: {len(examples)} ok, {fail} failed, {too_long} too long', flush=True)

    print(f'  Done: {len(examples)} examples, {fail} failed, {too_long} too long', flush=True)

    total_blocks = sum(e['n_blocks'] for e in examples)
    total_main = sum(e['n_main'] for e in examples)
    total_other = total_blocks - total_main
    print(f'  Blocks: {total_blocks:,} (main={total_main:,} [{total_main/total_blocks*100:.1f}%], other={total_other:,} [{total_other/total_blocks*100:.1f}%])', flush=True)
    print(f'  Avg blocks/page: {total_blocks/len(examples):.1f}', flush=True)
    print(f'  Avg tokens/page: {sum(len(e["input_ids"]) for e in examples)/len(examples):.0f}', flush=True)

    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != IGNORE_LABEL
    flat_preds = preds[mask]
    flat_labels = labels[mask]

    acc = (flat_preds == flat_labels).mean()
    f1 = f1_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)
    prec = precision_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)
    rec = recall_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }


def html_to_text(html_str):
    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    pred_ngrams = Counter(ngrams(pred_tokens, n))
    if not ref_ngrams or not pred_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & pred_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def insert_block_markers_inference(simplified_html):
    """Insert [BLOCK] tokens for inference (no labels). Returns (marked_html, item_ids)."""
    pattern = re.compile(r'(_item_id="(\d+)")')
    item_ids = []
    parts = []
    last_end = 0
    for m in pattern.finditer(simplified_html):
        item_id = m.group(2)
        parts.append(simplified_html[last_end:m.start()])
        parts.append(BLOCK_TOKEN + ' ')
        parts.append(m.group(0))
        last_end = m.end()
        item_ids.append(item_id)
    if not item_ids:
        return None, []
    parts.append(simplified_html[last_end:])
    return ''.join(parts), item_ids


@torch.no_grad()
def classify_page(model, tokenizer, block_token_id, simplified_html, device):
    """Classify blocks on a single page. Returns dict {item_id: 'main'/'other'}."""
    marked_html, item_ids = insert_block_markers_inference(simplified_html)
    if marked_html is None:
        return {}
    encoding = tokenizer(
        marked_html, truncation=True, max_length=MAX_LENGTH,
        add_special_tokens=True, padding=False, return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    outputs = model(input_ids=input_ids, attention_mask={'full_attention': None})
    logits = outputs.logits[0]
    block_positions = (input_ids[0] == block_token_id).nonzero(as_tuple=True)[0]
    preds = logits[block_positions].argmax(dim=-1).cpu().tolist()
    labels = {}
    for i, item_id in enumerate(item_ids):
        if i < len(preds):
            labels[item_id] = 'main' if preds[i] == LABEL_MAIN else 'other'
        else:
            labels[item_id] = 'other'
    return labels


def eval_rouge5_on_wmb(model, tokenizer, block_token_id, device, wmb_records):
    """Run end-to-end ROUGE-5 eval on WMB sample. Returns (avg_rouge5, per_level_dict)."""
    model.eval()
    scores = []
    scores_by_level = {}
    empty = 0

    for rec in wmb_records:
        html_content = rec.get('html', '')
        reference = rec.get('convert_main_content', '')
        level = rec.get('meta', {}).get('level', 'unknown')

        if not html_content or not reference:
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        try:
            simplified, map_html = simplify_html(html_content)
        except Exception:
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        labels = classify_page(model, tokenizer, block_token_id, simplified, device)

        n_main = sum(1 for v in labels.values() if v == 'main')
        if n_main == 0:
            pred_text = ''
        else:
            try:
                main_html = extract_main_html(map_html, labels)
                pred_text = html_to_text(main_html).strip()
            except Exception:
                pred_text = ''

        if not pred_text:
            empty += 1
            r5 = 0.0
        else:
            r5 = rouge_n_f1(reference, pred_text, n=5)

        scores.append(r5)
        scores_by_level.setdefault(level, []).append(r5)

    avg = sum(scores) / max(len(scores), 1)
    level_avgs = {}
    for lev in ['simple', 'mid', 'hard']:
        vals = scores_by_level.get(lev, [])
        level_avgs[lev] = sum(vals) / max(len(vals), 1)

    return avg, level_avgs, empty


class RougeEvalCallback(TrainerCallback):
    """Runs end-to-end ROUGE-5 on WMB sample after each epoch (main process only)."""

    def __init__(self, tokenizer, block_token_id, wmb_records):
        self.tokenizer = tokenizer
        self.block_token_id = block_token_id
        self.wmb_records = wmb_records

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.local_rank > 0:
            return
        if not self.wmb_records:
            return

        device = next(model.parameters()).device
        avg, level_avgs, empty = eval_rouge5_on_wmb(
            model, self.tokenizer, self.block_token_id, device, self.wmb_records,
        )
        print(f'\n  WMB ROUGE-5 (epoch {state.epoch:.0f}, {len(self.wmb_records)} pages):  '
              f'All={avg:.4f}  Simple={level_avgs["simple"]:.4f}  '
              f'Mid={level_avgs["mid"]:.4f}  Hard={level_avgs["hard"]:.4f}  '
              f'Empty={empty}', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='HF model name')
    parser.add_argument('--limit', type=int, default=0, help='Limit pages for testing')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--val-pages', type=int, default=1000)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--original-only', action='store_true', help='Use only original 15K data')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    args = parser.parse_args()

    model_name = args.model
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_short = model_name.split('/')[-1].replace('Qwen3-Embedding-', '')
        output_dir = os.path.join(SCRIPT_DIR, f'block_classifier_{model_short}')

    set_seed(SEED)

    # Check if launched via accelerate (multi-GPU)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_main = local_rank <= 0

    if is_main:
        print(f'Loading tokenizer: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add [BLOCK] special token
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': [BLOCK_TOKEN]})
    block_token_id = tokenizer.convert_tokens_to_ids(BLOCK_TOKEN)
    if is_main:
        print(f'  Added {num_added} special tokens. [BLOCK] id = {block_token_id}', flush=True)

    # Load WMB eval sample for end-to-end ROUGE-5 monitoring
    wmb_records = []
    if os.path.exists(WMB_EVAL_PATH):
        with open(WMB_EVAL_PATH) as f:
            for line in f:
                wmb_records.append(json.loads(line))
        if is_main:
            print(f'  Loaded {len(wmb_records)} WMB eval pages for ROUGE-5 monitoring', flush=True)

    # Prepare data (all processes do this to avoid needing to broadcast)
    if is_main:
        print('\nPreparing training examples from CC...', flush=True)
    examples = prepare_examples(tokenizer, block_token_id, limit=args.limit,
                                max_length=args.max_length, original_only=args.original_only)

    # Split train/val
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(examples))
    val_size = min(args.val_pages, len(examples) // 5)
    val_idx = set(indices[:val_size])

    train_examples = [e for i, e in enumerate(examples) if i not in val_idx]
    val_examples = [e for i, e in enumerate(examples) if i in val_idx]
    if is_main:
        print(f'\n  Train: {len(train_examples)} pages', flush=True)
        print(f'  Val:   {len(val_examples)} pages', flush=True)

    # Compute class weights
    train_main = sum(e['n_main'] for e in train_examples)
    train_other = sum(e['n_blocks'] - e['n_main'] for e in train_examples)
    train_total = train_main + train_other
    w_other = train_total / (2 * train_other) if train_other > 0 else 1.0
    w_main = train_total / (2 * train_main) if train_main > 0 else 1.0
    class_weights = [w_other, w_main]
    if is_main:
        print(f'  Class weights: other={w_other:.3f}, main={w_main:.3f}', flush=True)

    # Load model with SDPA
    if is_main:
        print(f'\nLoading model: {model_name} (SDPA, bidirectional)', flush=True)
        print(f'  Output dir: {output_dir}', flush=True)
        if args.gradient_checkpointing:
            print(f'  Gradient checkpointing: enabled', flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model.resize_token_embeddings(len(tokenizer))
    make_model_bidirectional(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='steps',
        eval_steps=0.25,
        save_strategy='steps',
        save_steps=0.25,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to='none',
        ddp_find_unused_parameters=False,
    )

    callbacks = []
    if wmb_records:
        callbacks.append(RougeEvalCallback(tokenizer, block_token_id, wmb_records))

    trainer = BidirectionalTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=BlockDataset(train_examples),
        eval_dataset=BlockDataset(val_examples),
        data_collator=BlockCollator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    if is_main:
        print('\nTraining...', flush=True)
    t0 = time.time()
    trainer.train()
    if is_main:
        print(f'\nTraining took {time.time() - t0:.0f}s', flush=True)

    # Final eval
    if is_main:
        print('\nFinal evaluation:', flush=True)
    metrics = trainer.evaluate()
    if is_main:
        for k, v in sorted(metrics.items()):
            print(f'  {k}: {v}', flush=True)

    # Save
    if is_main:
        trainer.save_model(os.path.join(output_dir, 'final'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'final'))
        print(f'\nModel saved to {output_dir}/final', flush=True)


if __name__ == '__main__':
    main()
