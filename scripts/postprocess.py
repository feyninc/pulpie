"""Stage 3: CPU postprocess — reconstruct HTML + convert to markdown.

Usage:
    python scripts/postprocess.py --input-dir /data/classified/ --out-dir /data/output/ --workers 4

Reads classified .pt files, reconstructs main-content HTML, converts to markdown.
Writes output as JSONL with page_id, labels, html, markdown per line.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from pulpie.reconstruct import extract_main_html


def _postprocess_page(page: dict) -> dict:
    """Reconstruct HTML and convert to markdown for a single page."""
    if page.get("error") or not page.get("labels"):
        return {
            "page_id": page["page_id"],
            "labels": page.get("labels", {}),
            "html": "",
            "markdown": "",
            "error": page.get("error"),
        }

    main_html = extract_main_html(page["map_html"], page["labels"])

    try:
        import html2text

        h = html2text.HTML2Text(bodywidth=0)
        h.ignore_links = False
        h.ignore_images = False
        markdown = h.handle(main_html).strip()
    except ImportError:
        markdown = main_html

    return {
        "page_id": page["page_id"],
        "labels": page["labels"],
        "html": main_html,
        "markdown": markdown,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Postprocess classified pages to markdown")
    parser.add_argument("--input-dir", required=True, help="Directory with classified .pt files")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSONL results")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU workers")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(input_dir.glob("batch_*.pt"))
    print(f"Found {len(batch_files)} classified batch files")

    t0 = time.perf_counter()
    total_pages = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for batch_file in batch_files:
            pages = torch.load(batch_file, weights_only=False)
            n_pages = len(pages)

            futures = {executor.submit(_postprocess_page, page): i for i, page in enumerate(pages)}
            results = [None] * n_pages
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()

            out_file = out_dir / batch_file.with_suffix(".jsonl").name
            with open(out_file, "w") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            total_pages += n_pages
            elapsed = time.perf_counter() - t0
            pps = total_pages / elapsed
            print(f"  {batch_file.name} → {out_file.name}: {n_pages} pages ({pps:.0f} pps)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total_pages} pages in {elapsed:.1f}s ({total_pages/elapsed:.0f} pps)")


if __name__ == "__main__":
    main()
