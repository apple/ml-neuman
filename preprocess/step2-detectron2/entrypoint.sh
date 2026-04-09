#!/bin/bash
set -e

MODE="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"
EXTRA_ARGS=("${@:4}")

if [ -z "$MODE" ] || [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: entrypoint.sh <masks|densepose> <input_dir> <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ "$MODE" = "masks" ]; then
    echo "=== Running instance segmentation ==="
    python /app/run_masks.py "$INPUT_DIR" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"

elif [ "$MODE" = "densepose" ]; then
    echo "=== Running DensePose ==="
    python /app/run_densepose.py "$INPUT_DIR" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
else
    echo "Unknown mode: $MODE (use 'masks' or 'densepose')"
    exit 1
fi
