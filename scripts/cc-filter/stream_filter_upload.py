"""Stream WARCs from CC-MAIN-2026-12, filter English HTML, upload to HF bucket.

Per WARC: download → extract → filter → zstd-compress JSONL → upload → delete local.
Resumable: skips WARCs whose output already exists in the bucket.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

import fasttext
import requests
import zstandard as zstd
from huggingface_hub import batch_bucket_files, list_bucket_tree
from warcio.archiveiterator import ArchiveIterator

# ── Config ──
CC_BASE = "https://data.commoncrawl.org/"
CRAWL = "CC-MAIN-2026-12"
BUCKET = "chonkie-ai/cc-main-2026-12-en"
REMOTE_DIR = "warc"  # hf://buckets/<bucket>/warc/warc_00000.jsonl.zst

DATA_DIR = os.environ.get(
    "CC_FILTER_DATA_DIR",
    os.path.join(os.path.expanduser("~"), "workspace", "cc_data"),
)
WARC_PATHS_FILE = os.path.join(DATA_DIR, "warc.paths")
LID_MODEL = os.path.join(DATA_DIR, "lid.176.bin")
LOCAL_STAGING = os.path.join(DATA_DIR, "staging")

MIN_HTML_BYTES = 2_000
MAX_HTML_BYTES = 500_000
MIN_TEXT_LEN = 200
LANG_CONFIDENCE = 0.8
TARGET_LANG = "en"

os.makedirs(LOCAL_STAGING, exist_ok=True)

_LID = None


def get_lid():
    global _LID
    if _LID is None:
        # Suppress fasttext warning about loading binary
        fasttext.FastText.eprint = lambda *a, **k: None
        _LID = fasttext.load_model(LID_MODEL)
    return _LID


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SCRIPT_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def extract_visible_text(html: str) -> str:
    html = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text).strip()
    return text[:5000]


def process_warc(idx: int, warc_path: str) -> dict:
    """Download one WARC, filter English, write zst, upload to bucket, delete local.

    Returns a stats dict.
    """
    url = CC_BASE + warc_path
    local_warc = os.path.join(LOCAL_STAGING, f"warc_{idx:05d}.warc.gz")
    local_out = os.path.join(LOCAL_STAGING, f"warc_{idx:05d}.jsonl.zst")
    remote_path = f"{REMOTE_DIR}/warc_{idx:05d}.jsonl.zst"

    stats = {
        "idx": idx,
        "warc_path": warc_path,
        "ok": False,
        "records": 0,
        "html": 0,
        "en": 0,
        "kept_bytes": 0,
        "t_download": 0.0,
        "t_filter": 0.0,
        "t_upload": 0.0,
        "error": None,
    }

    try:
        # ── Download ──
        t0 = time.perf_counter()
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_warc, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        stats["t_download"] = time.perf_counter() - t0

        # ── Filter ──
        t0 = time.perf_counter()
        lid = get_lid()
        cctx = zstd.ZstdCompressor(level=10)
        kept = 0
        bytes_out = 0
        with open(local_out, "wb") as raw_out, cctx.stream_writer(raw_out) as z:
            with open(local_warc, "rb") as f:
                for record in ArchiveIterator(f):
                    if record.rec_type != "response":
                        continue
                    stats["records"] += 1
                    url_ = record.rec_headers.get_header("WARC-Target-URI") or ""
                    ct = (
                        record.http_headers.get_header("Content-Type")
                        if record.http_headers else ""
                    ) or ""
                    status = (
                        record.http_headers.get_statuscode()
                        if record.http_headers else ""
                    )
                    if str(status) != "200" or "text/html" not in ct.lower():
                        continue
                    content = record.content_stream().read()
                    n = len(content)
                    if n < MIN_HTML_BYTES or n > MAX_HTML_BYTES:
                        continue
                    stats["html"] += 1
                    try:
                        html = content.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    text = extract_visible_text(html)
                    if len(text) < MIN_TEXT_LEN:
                        continue
                    pred = lid.predict(text.replace("\n", " ")[:2000])
                    lang = pred[0][0].replace("__label__", "")
                    conf = float(pred[1][0])
                    if lang != TARGET_LANG or conf < LANG_CONFIDENCE:
                        continue
                    stats["en"] += 1
                    try:
                        domain = urlparse(url_).netloc.lower()
                    except Exception:
                        domain = ""
                    rec = {
                        "url": url_,
                        "domain": domain,
                        "html": html,
                        "html_bytes": n,
                        "lang_conf": round(conf, 3),
                    }
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    z.write(line.encode("utf-8"))
                    kept += 1
        stats["t_filter"] = time.perf_counter() - t0
        stats["kept_bytes"] = os.path.getsize(local_out)

        # ── Upload ──
        t0 = time.perf_counter()
        batch_bucket_files(
            BUCKET,
            add=[(local_out, remote_path)],
            token=os.environ["HF_TOKEN"],
        )
        stats["t_upload"] = time.perf_counter() - t0

        stats["ok"] = True

    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)

    finally:
        # Always clean up local files
        for p in (local_warc, local_out):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    return stats


def list_existing_remote() -> set[int]:
    """Return the set of already-uploaded warc indices in the bucket."""
    done = set()
    try:
        for entry in list_bucket_tree(
            BUCKET,
            prefix=REMOTE_DIR,
            recursive=True,
            token=os.environ["HF_TOKEN"],
        ):
            name = getattr(entry, "path", None) or getattr(entry, "name", None) or str(entry)
            m = re.search(r"warc_(\d{5})\.jsonl\.zst", name)
            if m:
                done.add(int(m.group(1)))
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            return set()
        raise
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None,
                        help="Exclusive end idx; None = all WARCs.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-warcs", type=int, default=None,
                        help="Safety cap: stop after this many WARCs processed.")
    args = parser.parse_args()

    with open(WARC_PATHS_FILE) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    end = args.end if args.end is not None else len(paths)
    indices = list(range(args.start, end))

    print(f"Crawl: {CRAWL}")
    print(f"Bucket: hf://buckets/{BUCKET}")
    print(f"WARCs requested: [{args.start}, {end}) = {len(indices)} total")
    print(f"Workers: {args.workers}")
    print(f"LID threshold: {LANG_CONFIDENCE}")

    print("Checking what's already in the bucket...")
    existing = list_existing_remote()
    print(f"  {len(existing)} WARCs already uploaded → skipping those")

    todo = [i for i in indices if i not in existing]
    if args.max_warcs:
        todo = todo[: args.max_warcs]
    print(f"  Will process {len(todo)} WARCs")
    if not todo:
        print("Nothing to do.")
        return

    t_start = time.perf_counter()
    total_en = 0
    total_bytes = 0
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_warc, i, paths[i]): i for i in todo
        }
        for n_done, fut in enumerate(as_completed(futures), 1):
            s = fut.result()
            if s["ok"]:
                n_ok += 1
                total_en += s["en"]
                total_bytes += s["kept_bytes"]
            else:
                n_fail += 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / n_done * (len(todo) - n_done)
            avg_pps = total_en / elapsed if elapsed > 0 else 0
            print(
                f"[{n_done:>4}/{len(todo)}] "
                f"idx={s['idx']:>5} "
                f"ok={s['ok']} "
                f"en={s['en']:>5} "
                f"html={s['html']:>5} "
                f"size={s['kept_bytes']/1e6:>5.1f}MB "
                f"dl={s['t_download']:>4.1f}s "
                f"filt={s['t_filter']:>4.1f}s "
                f"up={s['t_upload']:>4.1f}s "
                f"| total: en={total_en:,} gb={total_bytes/1e9:.2f} "
                f"avg={avg_pps:.1f}pg/s eta={eta/60:.0f}m"
                + (f" ERROR: {s['error']}" if s['error'] else ""),
                flush=True,
            )

    print()
    print(f"Done. ok={n_ok} fail={n_fail} total_en={total_en:,} "
          f"size={total_bytes/1e9:.2f} GB elapsed={(time.perf_counter()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
