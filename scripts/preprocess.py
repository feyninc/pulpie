"""Stage 1: CPU preprocess — simplify HTML + tokenize + chunk, write batch files.

Usage:
    python scripts/preprocess.py --input pages.jsonl --out-dir /data/batches/ --batch-size 500 --workers 8

Input: JSONL file with {"html": "...", "url": "...", ...} per line.
Output: batch_0000.pt, batch_0001.pt, ... in out-dir.
Each .pt file contains a list of prepared pages (chunks + metadata).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from pulpie.chunker import SEP_TOKEN, extract_blocks, pack_chunks, tokenize_blocks
from pulpie.model_utils import extract_item_ids, resolve_model_id
from pulpie.simplify import simplify

_worker_tokenizer = None
_worker_sep_token_id = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer, _worker_sep_token_id  # noqa: PLW0603
    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if SEP_TOKEN not in _worker_tokenizer.get_vocab():
        _worker_tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})
    _worker_sep_token_id = _worker_tokenizer.convert_tokens_to_ids(SEP_TOKEN)


def _prepare_page(html: str, page_id: int, max_tokens: int, cutoff_length: int) -> dict:
    try:
        simplified, map_html = simplify(html, cutoff_length=cutoff_length)
    except Exception as e:
        return {"page_id": page_id, "error": str(e), "chunks": [], "item_ids": [], "n_blocks": 0, "map_html": ""}

    blocks = extract_blocks(simplified)
    if not blocks:
        return {"page_id": page_id, "error": None, "chunks": [], "item_ids": [], "n_blocks": 0, "map_html": map_html}

    item_ids = extract_item_ids(blocks)
    assert _worker_tokenizer is not None
    block_token_ids = tokenize_blocks(blocks, _worker_tokenizer)
    chunks = pack_chunks(
        block_token_ids,
        max_tokens=max_tokens,
        sep_token_id=_worker_sep_token_id,
        bos_token_id=_worker_tokenizer.bos_token_id,
        eos_token_id=_worker_tokenizer.eos_token_id,
    )

    return {
        "page_id": page_id,
        "error": None,
        "chunks": chunks,
        "item_ids": item_ids,
        "n_blocks": len(blocks),
        "map_html": map_html,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess HTML pages into batch files")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--out-dir", required=True, help="Output directory for batch files")
    parser.add_argument("--model", default="orange-small", help="Model name (for tokenizer)")
    parser.add_argument("--batch-size", type=int, default=500, help="Pages per batch file")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per chunk")
    parser.add_argument("--cutoff-length", type=int, default=500, help="Text truncation length")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args.model)

    # Read all pages
    print(f"Reading {args.input}...")
    pages = []
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            pages.append(rec.get("html", ""))
    print(f"  {len(pages)} pages")

    # Process in batches
    t0 = time.perf_counter()
    batch_idx = 0
    total_processed = 0
    total_chunks = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(model_id,)) as executor:
        for batch_start in range(0, len(pages), args.batch_size):
            batch_pages = pages[batch_start : batch_start + args.batch_size]

            futures = {
                executor.submit(_prepare_page, html, batch_start + i, args.max_tokens, args.cutoff_length): i
                for i, html in enumerate(batch_pages)
            }

            prepared = [None] * len(batch_pages)
            for future in as_completed(futures):
                i = futures[future]
                prepared[i] = future.result()

            n_chunks = sum(len(p["chunks"]) for p in prepared)
            total_chunks += n_chunks
            total_processed += len(batch_pages)

            batch_file = out_dir / f"batch_{batch_idx:06d}.pt"
            tmp_file = out_dir / f"batch_{batch_idx:06d}.tmp"
            torch.save(prepared, tmp_file)
            os.rename(tmp_file, batch_file)

            elapsed = time.perf_counter() - t0
            pps = total_processed / elapsed
            print(f"  batch {batch_idx}: {len(batch_pages)} pages, {n_chunks} chunks → {batch_file.name} ({pps:.0f} pps)")
            batch_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total_processed} pages, {total_chunks} chunks, {batch_idx} batches in {elapsed:.1f}s ({total_processed/elapsed:.0f} pps)")


if __name__ == "__main__":
    main()
