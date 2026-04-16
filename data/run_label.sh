#!/bin/bash
# Label CC pages with DeepSeek V3.2 via AWS Bedrock.
# Run inside tmux/screen — this takes ~15 hours.
#
# Usage:
#   ./run_label.sh              # label all pages
#   ./run_label.sh --limit 100  # test with 100 pages
#
# Resume: just re-run. Already-labeled URLs are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/.logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/label_cc_${TIMESTAMP}.log"

echo "=================================================="
echo "  CC Page Labeling — DeepSeek V3.2 on Bedrock"
echo "=================================================="
echo "  Start:      $(date)"
echo "  Log:        ${LOG_FILE}"
echo "  Input:      ${SCRIPT_DIR}/cc_sampled.jsonl"
echo "  Output:     ${SCRIPT_DIR}/cc_labeled.jsonl"
echo "  Concurrency: 20"
echo "=================================================="
echo ""

python3 "${SCRIPT_DIR}/label_cc.py" \
    --concurrency 20 \
    --input "${SCRIPT_DIR}/cc_sampled.jsonl" \
    --output "${SCRIPT_DIR}/cc_labeled.jsonl" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Done at $(date). Log: ${LOG_FILE}"
