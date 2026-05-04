"""Orchestrator: stream shards from HF bucket → process → upload.

Downloads one shard at a time, runs preprocess → GPU infer → postprocess,
uploads cleaned output, moves to next shard. Tracks progress via done file
so it can resume after interruption.

Usage:
    python scripts/run_bucket.py \
        --input-bucket chonkie-ai/cc-main-2026-12-en \
        --output-bucket chonkie-ai/cc-main-2026-12-en-clean \
        --model /path/to/model \
        --work-dir /tmp/pulpie_run
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path


def hf_cmd(*args: str) -> subprocess.CompletedProcess:
    hf_bin = os.path.expanduser("~/.local/bin/hf")
    return subprocess.run([hf_bin, *args], capture_output=True, text=True, timeout=300)


def list_shards(bucket: str) -> list[str]:
    result = hf_cmd("buckets", "ls", f"hf://buckets/{bucket}", "-R")
    shards = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.endswith(".jsonl.zst"):
            path = line.split()[-1]
            shards.append(path)
    return sorted(shards)


def download_shard(bucket: str, shard_path: str, local_path: Path) -> bool:
    result = hf_cmd("buckets", "cp", f"hf://buckets/{bucket}/{shard_path}", str(local_path))
    return result.returncode == 0


def upload_result(bucket: str, local_path: Path, remote_path: str) -> bool:
    result = hf_cmd("buckets", "cp", str(local_path), f"hf://buckets/{bucket}/{remote_path}")
    return result.returncode == 0


def decompress_shard(zst_path: Path, jsonl_path: Path) -> int:
    import zstandard

    n = 0
    with open(zst_path, "rb") as fin, open(jsonl_path, "w") as fout:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(fin)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            fout.write(line)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Stream shards from HF bucket, process, upload")
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", default="/tmp/pulpie_run")
    parser.add_argument("--pre-workers", type=int, default=12)
    parser.add_argument("--post-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--start", type=int, default=0, help="Start from shard index")
    parser.add_argument("--limit", type=int, default=0, help="Max shards to process (0=all)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    batch_dir = work_dir / "batches"
    class_dir = work_dir / "classified"
    output_dir = work_dir / "output"
    done_file = work_dir / "done.txt"

    for d in [work_dir, batch_dir, class_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    done_shards: set[str] = set()
    if done_file.exists():
        done_shards = {s for s in done_file.read_text().strip().split("\n") if s}

    print(f"Listing shards in {args.input_bucket}...")
    all_shards = list_shards(args.input_bucket)
    print(f"  {len(all_shards)} shards found, {len(done_shards)} already done")

    shards = [s for s in all_shards if s not in done_shards]
    shards = shards[args.start:]
    if args.limit > 0:
        shards = shards[: args.limit]
    print(f"  Processing {len(shards)} shards")

    scripts_dir = Path(__file__).parent
    n_gpus = int(subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]
    ).strip())
    print(f"  GPUs: {n_gpus}")

    total_pages = 0
    t_start = time.perf_counter()

    for shard_idx, shard_path in enumerate(shards):
        shard_name = Path(shard_path).stem.replace(".jsonl", "")
        t0 = time.perf_counter()

        # Clean work dirs
        for d in [batch_dir, class_dir, output_dir]:
            for f in d.glob("*"):
                f.unlink()

        # Download
        zst_local = work_dir / f"{shard_name}.jsonl.zst"
        jsonl_local = work_dir / f"{shard_name}.jsonl"
        print(f"\n[{shard_idx+1}/{len(shards)}] {shard_path}")
        print(f"  Downloading...", end=" ", flush=True)
        if not download_shard(args.input_bucket, shard_path, zst_local):
            print("FAILED")
            continue
        print("OK")

        # Decompress
        print(f"  Decompressing...", end=" ", flush=True)
        n_pages = decompress_shard(zst_local, jsonl_local)
        print(f"{n_pages} pages")
        zst_local.unlink()

        # Stage 1: preprocess
        print(f"  Preprocessing...", end=" ", flush=True)
        t1 = time.perf_counter()
        subprocess.run(
            [
                sys.executable, str(scripts_dir / "preprocess.py"),
                "--input", str(jsonl_local),
                "--out-dir", str(batch_dir),
                "--model", args.model,
                "--batch-size", str(args.batch_size),
                "--workers", str(args.pre_workers),
            ],
            capture_output=True, timeout=600,
        )
        t2 = time.perf_counter()
        print(f"{t2-t1:.1f}s")
        jsonl_local.unlink()

        # Stage 2: GPU inference (all GPUs, fresh processes)
        print(f"  GPU inference ({n_gpus} GPUs)...", end=" ", flush=True)
        gpu_procs = []
        for i in range(n_gpus):
            p = subprocess.Popen(
                [
                    sys.executable, str(scripts_dir / "infer.py"),
                    "--batch-dir", str(batch_dir),
                    "--out-dir", str(class_dir),
                    "--model", args.model,
                    "--device", f"cuda:{i}",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            gpu_procs.append(p)
        for p in gpu_procs:
            p.wait()
        t3 = time.perf_counter()
        print(f"{t3-t2:.1f}s")

        # Stage 3: postprocess
        print(f"  Postprocessing...", end=" ", flush=True)
        subprocess.run(
            [
                sys.executable, str(scripts_dir / "postprocess.py"),
                "--input-dir", str(class_dir),
                "--out-dir", str(output_dir),
                "--workers", str(args.post_workers),
            ],
            capture_output=True, timeout=600,
        )
        t4 = time.perf_counter()
        print(f"{t4-t3:.1f}s")

        # Upload results
        print(f"  Uploading...", end=" ", flush=True)
        for out_file in sorted(output_dir.glob("*.jsonl")):
            upload_result(
                args.output_bucket,
                out_file,
                f"clean/{shard_name}/{out_file.name}",
            )
        print("OK")

        elapsed = time.perf_counter() - t0
        total_pages += n_pages
        total_elapsed = time.perf_counter() - t_start
        pps = total_pages / total_elapsed

        print(f"  {n_pages} pages in {elapsed:.1f}s ({n_pages/elapsed:.0f} pps this shard, {pps:.0f} pps overall)")

        with open(done_file, "a") as f:
            f.write(shard_path + "\n")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Done: {total_pages:,} pages in {len(shards)} shards, {total_elapsed:.0f}s ({total_pages/max(total_elapsed,1):.0f} pps)")


if __name__ == "__main__":
    main()
