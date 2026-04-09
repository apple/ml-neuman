#!/bin/bash
set -e

MODE="$1"       # align | optimize | all
SCENE_DIR="$2"  # e.g., /data/my_video/output

if [ -z "$MODE" ] || [ -z "$SCENE_DIR" ]; then
    echo "Usage: entrypoint.sh <align|optimize|all> <scene_dir>"
    echo "  scene_dir should contain: sparse/, images/, smpl_pred/, segmentations/"
    exit 1
fi

cd /neuman/preprocess

if [ "$MODE" = "align" ] || [ "$MODE" = "all" ]; then
    echo "=== Step 9: Solve scale ambiguity ==="
    python export_alignment.py \
        --scene_dir "$SCENE_DIR/sparse" \
        --images_dir "$SCENE_DIR/images" \
        --raw_smpl "$SCENE_DIR/smpl_pred" \
        --smpl_estimator="romp"
fi

if [ "$MODE" = "optimize" ] || [ "$MODE" = "all" ]; then
    echo "=== Step 10: Optimize SMPL using silhouette ==="
    python optimize_smpl.py \
        --scene_dir "$SCENE_DIR"
fi

echo "=== Done ==="
