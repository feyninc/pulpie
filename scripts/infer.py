"""Stage 2: GPU inference — read batch files, classify, write results.

Usage:
    python scripts/infer.py --batch-dir /data/batches/ --out-dir /data/classified/ --device cuda:0

Multiple instances can run in parallel on different GPUs.
Each claims a batch file by atomic rename (.pt → .processing) to prevent duplicates.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from pulpie.chunker import SEP_TOKEN
from pulpie.model_utils import load_model_and_tokenizer, predictions_to_labels, resolve_model_id


def claim_batch(batch_dir: Path) -> Path | None:
    """Atomically claim the next unprocessed batch file."""
    for f in sorted(batch_dir.glob("batch_*.pt")):
        claimed = f.with_suffix(".processing")
        try:
            os.rename(f, claimed)
            return claimed
        except OSError:
            continue
    return None


@torch.no_grad()
def infer_batch(
    prepared: list[dict],
    model: torch.nn.Module,
    sep_token_id: int,
    device: torch.device,
    max_batch_tokens: int,
) -> list[dict]:
    """Run inference on a list of prepared pages, return results with labels."""
    pad_id = model.config.pad_token_id or 0

    # Flatten all chunks
    all_chunks: list[tuple[list[int], list[int], int]] = []
    for page_idx, page in enumerate(prepared):
        if page["error"] or not page["chunks"]:
            continue
        for chunk_ids, block_indices in page["chunks"]:
            all_chunks.append((chunk_ids, block_indices, page_idx))

    if not all_chunks:
        return [
            {"page_id": p["page_id"], "labels": {}, "map_html": p["map_html"], "error": p["error"]}
            for p in prepared
        ]

    # Sort by length
    all_chunks.sort(key=lambda x: len(x[0]))

    # Batched inference
    chunk_predictions: dict[int, list[tuple[list[int], list[int]]]] = {}
    i = 0
    while i < len(all_chunks):
        max_seq = len(all_chunks[min(i + 64, len(all_chunks) - 1)][0])
        bs = max(1, max_batch_tokens // max(max_seq, 1))
        batch = all_chunks[i : i + bs]
        i += bs

        max_len = max(len(c[0]) for c in batch)
        input_ids = []
        attention_mask = []
        for chunk_ids, _, _ in batch:
            pad_len = max_len - len(chunk_ids)
            input_ids.append(chunk_ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(chunk_ids) + [0] * pad_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t)

        for batch_idx, (_, block_indices, page_idx) in enumerate(batch):
            logits = outputs.logits[batch_idx]
            sep_positions = (input_ids_t[batch_idx] == sep_token_id).nonzero(as_tuple=True)[0]
            preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()
            if page_idx not in chunk_predictions:
                chunk_predictions[page_idx] = []
            chunk_predictions[page_idx].append((block_indices, preds))

    # Assemble results
    results = []
    for page_idx, page in enumerate(prepared):
        if page["error"] or not page["chunks"]:
            results.append({
                "page_id": page["page_id"],
                "labels": {},
                "map_html": page["map_html"],
                "error": page["error"],
            })
            continue

        predictions = [0] * page["n_blocks"]
        for block_indices, preds in chunk_predictions.get(page_idx, []):
            for idx, block_idx in enumerate(block_indices):
                if idx < len(preds):
                    predictions[block_idx] = preds[idx]

        labels = predictions_to_labels(page["item_ids"], predictions)
        results.append({
            "page_id": page["page_id"],
            "labels": labels,
            "map_html": page["map_html"],
            "error": None,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU inference on preprocessed batch files")
    parser.add_argument("--batch-dir", required=True, help="Directory with batch .pt files")
    parser.add_argument("--out-dir", required=True, help="Output directory for classified results")
    parser.add_argument("--model", default="orange-small", help="Model name or path")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--max-batch-tokens", type=int, default=16384, help="Max tokens per GPU batch")
    parser.add_argument("--persistent", action="store_true", help="Keep polling for new batches instead of exiting")
    parser.add_argument("--stop-file", default=None, help="Exit when this file appears (used with --persistent)")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_id = resolve_model_id(args.model)

    print(f"Loading model on {device}...")
    model, tokenizer, sep_token_id = load_model_and_tokenizer(model_id, device)
    print(f"  Ready. VRAM: {torch.cuda.memory_allocated(device) / 1e6:.0f}MB")

    total_pages = 0
    total_batches = 0
    t0 = time.perf_counter()

    stop_file = Path(args.stop_file) if args.stop_file else None

    while True:
        # Check for stop signal in persistent mode
        if stop_file and stop_file.exists():
            print(f"[{args.device}] Stop file detected, exiting.", flush=True)
            break

        batch_file = claim_batch(batch_dir)
        if batch_file is None:
            if args.persistent:
                time.sleep(0.1)
                continue
            # Non-persistent: exit when no work left
            processing = list(batch_dir.glob("batch_*.processing"))
            remaining = list(batch_dir.glob("batch_*.pt"))
            if not processing and not remaining:
                break
            time.sleep(0.5)
            continue

        try:
            prepared = torch.load(batch_file, weights_only=False)
        except Exception as e:
            print(f"[{args.device}] Failed to load {batch_file.name}: {e}", flush=True)
            # Put it back for retry
            try:
                os.rename(batch_file, batch_file.with_suffix(".pt"))
            except OSError:
                batch_file.unlink(missing_ok=True)
            time.sleep(0.5)
            continue

        n_pages = len(prepared)
        n_chunks = sum(len(p["chunks"]) for p in prepared if not p["error"])

        results = infer_batch(prepared, model, sep_token_id, device, args.max_batch_tokens)

        # Write results (without chunks to save space)
        out_file = out_dir / batch_file.stem.replace(".processing", "").split(".")[0]
        out_file = out_dir / f"{batch_file.stem.split('.')[0]}.pt"
        torch.save(results, out_file)

        # Remove claimed file
        batch_file.unlink()

        total_pages += n_pages
        total_batches += 1
        elapsed = time.perf_counter() - t0
        pps = total_pages / elapsed
        print(f"  [{args.device}] batch {total_batches}: {n_pages} pages, {n_chunks} chunks → {out_file.name} ({pps:.1f} pps)")

    elapsed = time.perf_counter() - t0
    if total_pages > 0:
        print(f"\n[{args.device}] Done: {total_pages} pages in {total_batches} batches, {elapsed:.1f}s ({total_pages/elapsed:.1f} pps)")
    else:
        print(f"\n[{args.device}] No batches to process.")


if __name__ == "__main__":
    main()
