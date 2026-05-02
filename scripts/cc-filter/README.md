# CC-MAIN-2026-12 English filter pipeline

Stream Common Crawl WARC files, filter to English HTML (fasttext LID ≥ 0.8), and upload zstd-compressed JSONL to an HF Storage Bucket.

## Files

- `stream_filter_upload.py` — the pipeline. Per WARC: download → parse → filter → zstd → upload → delete local. Resumable (skips WARCs already in the bucket).
- `RECREATE_SETUP.md` — full box setup guide with pinned versions, troubleshooting, and sharding notes.
- `AGENT_PROMPT.md` — self-contained prompt for handing the setup task to an agent on a fresh VM.

## Scope

Source crawl: **CC-MAIN-2026-12** (March 2026, 100,000 WARCs).
Target bucket: `chonkie-ai/cc-main-2026-12-en` (private).

Expected output: ~290M English pages, ~5 TB zst JSONL.

## Best box

`c2-standard-16` spot — 16 vCPU matches the worker sweet spot exactly, no GPU needed. Compute cost for the full run is ~$40 on spot; egress to HF (~5 TB × ~$0.10/GB) dominates at ~$450.

Worker sweep on 16 WARCs showed:

| Workers | Wall | WARCs/min |
|---------|------|-----------|
| 4 | 356 s | 2.7 |
| 8 | 180 s | 5.3 |
| **16** | **92 s** | **10.4** |
| 32 | 94 s | 10.2 |

Past 16, CC's S3 per-connection latency is the bottleneck and additional workers don't help.
