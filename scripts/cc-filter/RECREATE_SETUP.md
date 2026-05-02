# Recreate CC filter pipeline on a new box

For an agent (or human) tasked with running the Common Crawl → English-only JSONL → HF bucket pipeline on a fresh GCP VM. The current run is on a GPU box that's overkill; this guide is for moving it to cheap CPU-only spot instances.

## What this pipeline does

Per WARC file in CC-MAIN-2026-12 (100,000 files, ~780 MB each):

1. Stream-download from `data.commoncrawl.org`
2. Parse WARC records, keep status=200 + `text/html`, size 2–500 KB
3. Detect language with fasttext `lid.176.bin`, keep English with confidence ≥ 0.8
4. Write JSONL (`{url, domain, html, html_bytes, lang_conf}`) compressed with zstd
5. Upload to HF bucket `chonkie-ai/cc-main-2026-12-en` at `warc/warc_{idx:05d}.jsonl.zst`
6. Delete local files

**Resumable**: the script lists the bucket at startup and skips WARCs already uploaded. Safe to kill/restart or run across multiple boxes at the same time as long as they don't double-process the same indices (use `--start`/`--end` ranges to shard).

## Target box

**Recommended**: `c2-standard-16` spot in `us-central1` or `us-east1` (close to HF's CDN POPs).

- 16 vCPU matches the worker sweet spot exactly (tested: 4→8→16 nearly linear, 32 no improvement — CC S3 + per-request latency is the bottleneck past 16)
- 64 GB RAM is plenty (actual usage peaked at ~6 GB)
- Need ~2 GB disk for staging (one WARC + one output) + ~200 MB for the LID model
- **No GPU needed**
- Spot preemption is safe: restart the script, it skips done WARCs

**Egress is the real cost**, not compute. GCP → HF public internet egress is ~$0.08–0.12/GB. Full crawl = ~5 TB out = **~$450 egress regardless of instance type**. Compute on `c2-standard-16` spot is ~$40 for the whole 6.7-day run.

## Setup steps (run on the new box)

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev g++ curl
```

`g++` is needed only if you install the original `fasttext` package (don't — see below). Safe to skip if only using `fasttext-numpy2`.

### 2. Python environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 "$HOME/.venv"
```

### 3. Python packages

```bash
uv pip install --python "$HOME/.venv/bin/python" \
    "huggingface_hub==1.12.2" \
    "warcio==1.8.1" \
    "zstandard==0.25.0" \
    "fasttext-numpy2==0.10.4" \
    "numpy==2.2.6" \
    "requests==2.33.1"
```

**Version notes** (learned the hard way):

- Use **`fasttext-numpy2`**, NOT `fasttext`. The original `fasttext==0.9.3` calls `np.array(..., copy=False)` which numpy 2 forbids. The `fasttext-numpy2` fork is a drop-in replacement that works with numpy 2.
- Need **`huggingface_hub>=1.0`** for the bucket APIs (`batch_bucket_files`, `list_bucket_tree`, `create_bucket`). The `hf buckets` CLI also needs this.
- Do NOT install `transformers` on the filter box — it doesn't need it, and older transformers conflicts with hf_hub 1.x.

### 4. Workspace

```bash
mkdir -p "$HOME/workspace/cc_data/staging"
cd "$HOME/workspace/cc_data"
```

### 5. Download fasttext LID model

```bash
curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
    -o "$HOME/workspace/cc_data/lid.176.bin"
```

(~126 MB)

### 6. Download WARC path listing

```bash
curl -sL https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/warc.paths.gz \
    -o warc.paths.gz
gunzip -k warc.paths.gz
wc -l warc.paths    # should be 100000
```

### 7. Copy the pipeline script

Put `stream_filter_upload.py` at `$HOME/workspace/cc_data/stream_filter_upload.py`. It's self-contained (reads `lid.176.bin`, `warc.paths`, writes to `staging/`, uploads to the hard-coded bucket `chonkie-ai/cc-main-2026-12-en`).

Source: same folder as this README on the original box. Copy it over with `gh`/scp/`hf buckets cp` or recreate the setup script from git if the project is committed.

### 8. HF authentication

```bash
export HF_TOKEN=hf_xxx   # needs write access to chonkie-ai buckets
```

Verify:

```bash
"$HOME/.venv/bin/python" -c "
import os
from huggingface_hub import bucket_info
print(bucket_info('chonkie-ai/cc-main-2026-12-en', token=os.environ['HF_TOKEN']))
"
```

### 9. Launch

```bash
# Foreground test on a small range first:
"$HOME/.venv/bin/python" stream_filter_upload.py --start 50000 --end 50005 --workers 16

# Full background run (your shard range — see sharding below):
nohup env HF_TOKEN="$HF_TOKEN" \
    "$HOME/.venv/bin/python" stream_filter_upload.py \
    --start 50000 --end 100000 --workers 16 \
    > run.log 2>&1 &
echo $! > run.pid
disown
```

## Sharding across multiple boxes

If you spin up N boxes to finish faster, split by index range so they don't clobber each other's bucket listings:

- Box A: `--start 0 --end 25000 --workers 16`
- Box B: `--start 25000 --end 50000 --workers 16`
- Box C: `--start 50000 --end 75000 --workers 16`
- Box D: `--start 75000 --end 100000 --workers 16`

The skip-already-uploaded logic means any overlap is safe but wasteful.

## Monitoring

```bash
# Last progress line (the script prints one per WARC on a single line):
grep "^\[" run.log | tail -1

# Pages/sec and ETA are in every line:
# [1234/99899] idx=1234 ok=True en=2970 ... | total: en=3,500,000 gb=57.4 avg=510pg/s eta=5600m

# Bucket size (run from anywhere with HF_TOKEN set):
"$HOME/.venv/bin/python" -c "
import os
from huggingface_hub import bucket_info
b = bucket_info('chonkie-ai/cc-main-2026-12-en', token=os.environ['HF_TOKEN'])
print(f'{b.total_files} files, {b.size/1e9:.1f} GB')
"
```

## Expected results

- **Pages/sec**: ~500 (sustained, 16 workers)
- **WARC wall time**: ~95 s (dl ~15s + filter ~75s + upload ~2s)
- **Full crawl**: ~6.7 days on one `c2-standard-16`, linear speedup with more boxes
- **Output size**: ~48 MB zst per WARC → ~5 TB total for 100K WARCs
- **English pages per WARC**: ~2,900 → ~290M pages total

## Troubleshooting

**`ValueError: Unable to avoid copy while creating an array as requested`**
You installed `fasttext` instead of `fasttext-numpy2`. Uninstall and reinstall the correct one.

**`TypeError: HfApi.list_bucket_tree() got an unexpected keyword argument 'path'`**
Your `huggingface_hub` is older than expected. The API uses `prefix=`. Upgrade to ≥1.12.

**Many download errors from CC**
Their S3 is occasionally flaky. The script wraps each WARC in try/except; failures are logged with `ok=False` and the next relaunch will retry them (they're not in the bucket). To retry failures specifically, just rerun the same `--start`/`--end` range.

**Running out of disk**
Staging only holds one WARC + one output per worker (~16 × ~830 MB = ~13 GB peak). If you see more, the script isn't deleting after upload — check the `finally` block in `process_warc()`.
