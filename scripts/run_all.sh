#!/bin/bash
# Run all three stages concurrently. Preprocess writes batches as they're ready,
# GPU workers pick them up, postprocess converts completed batches.
#
# Usage: bash scripts/run_all.sh <input.jsonl> <work_dir> <model_path> [n_pre_workers]

set -e

INPUT=$1
WORK_DIR=$2
MODEL=$3
N_PRE=${4:-12}

BATCH_DIR="$WORK_DIR/batches"
CLASS_DIR="$WORK_DIR/classified"
OUT_DIR="$WORK_DIR/output"

mkdir -p "$BATCH_DIR" "$CLASS_DIR" "$OUT_DIR"

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "=== Pulpie batch run ==="
echo "Input:      $INPUT"
echo "GPUs:       $N_GPUS"
echo "Pre workers: $N_PRE"
echo "Work dir:   $WORK_DIR"
echo ""

# Start timer
START_NS=$(date +%s%N)

# Stage 1: preprocess (background)
echo "[Stage 1] Starting preprocess..."
python3 scripts/preprocess.py \
    --input "$INPUT" \
    --out-dir "$BATCH_DIR" \
    --model "$MODEL" \
    --batch-size 250 \
    --workers "$N_PRE" &
PID_PRE=$!

# Give preprocess a 2s head start to write first batches
sleep 2

# Stage 2: GPU inference (one process per GPU, all background)
echo "[Stage 2] Starting $N_GPUS GPU workers..."
GPU_PIDS=""
for i in $(seq 0 $((N_GPUS - 1))); do
    python3 scripts/infer.py \
        --batch-dir "$BATCH_DIR" \
        --out-dir "$CLASS_DIR" \
        --device "cuda:$i" \
        --model "$MODEL" &
    GPU_PIDS="$GPU_PIDS $!"
done

# Stage 3: postprocess (background, polls for classified files)
echo "[Stage 3] Starting postprocess..."
(
    # Wait until at least one classified file exists
    while [ -z "$(ls $CLASS_DIR/batch_*.pt 2>/dev/null)" ]; do
        sleep 1
    done
    # Keep running until GPU workers are done and all files processed
    while true; do
        python3 scripts/postprocess.py \
            --input-dir "$CLASS_DIR" \
            --out-dir "$OUT_DIR" \
            --workers 4 2>/dev/null
        # Check if GPU workers are still alive
        ALL_DONE=true
        for pid in $GPU_PIDS; do
            if kill -0 $pid 2>/dev/null; then
                ALL_DONE=false
                break
            fi
        done
        if $ALL_DONE; then
            # One final pass
            python3 scripts/postprocess.py \
                --input-dir "$CLASS_DIR" \
                --out-dir "$OUT_DIR" \
                --workers 4 2>/dev/null
            break
        fi
        sleep 2
    done
) &
PID_POST=$!

# Wait for preprocess to finish
wait $PID_PRE
echo "[Stage 1] Preprocess done."

# Wait for all GPU workers
for pid in $GPU_PIDS; do
    wait $pid
done
echo "[Stage 2] All GPUs done."

# Wait for postprocess
wait $PID_POST
echo "[Stage 3] Postprocess done."

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
ELAPSED_S=$(echo "scale=1; $ELAPSED_MS / 1000" | bc)

N_PAGES=$(wc -l < "$INPUT")
PPS=$(echo "scale=1; $N_PAGES * 1000 / $ELAPSED_MS" | bc)

echo ""
echo "=== Summary ==="
echo "Pages:      $N_PAGES"
echo "Time:       ${ELAPSED_S}s"
echo "Throughput: ${PPS} pages/sec"
echo "Output:     $OUT_DIR/"
