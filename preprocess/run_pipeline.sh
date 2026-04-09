#!/bin/bash
# =============================================================================
# NeuMan Preprocessing Pipeline — Orchestration Script
#
# Runs all 7 containerized steps in sequence to preprocess a video for
# NeuMan / HUGS training.
#
# Usage:
#   ./run_pipeline.sh /path/to/my_video.mp4 /path/to/output_dir [/path/to/smplx]
#
# Prerequisites:
#   - Docker with NVIDIA GPU support (nvidia-container-toolkit)
#   - All 7 images built (run ./build_all.sh first)
# =============================================================================
set -e

VIDEO="$1"
DATA_DIR="$2"
SMPLX_DIR="${3:-./data/smplx}"

if [ -z "$VIDEO" ] || [ -z "$DATA_DIR" ]; then
    echo "Usage: ./run_pipeline.sh <video_path> <data_dir> [smplx_dir]"
    echo ""
    echo "  video_path  Path to input video (mp4, mov, etc.)"
    echo "  data_dir    Working directory for all intermediate and final outputs"
    echo "  smplx_dir   Path to SMPL-X model files (default: ./data/smplx)"
    echo ""
    echo "Output will be in <data_dir>/output/ in NeuMan format."
    exit 1
fi

VIDEO_ABS=$(realpath "$VIDEO")
DATA_ABS=$(realpath "$DATA_DIR")
SMPLX_ABS=$(realpath "$SMPLX_DIR")

VIDEO_NAME=$(basename "${VIDEO%.*}")

echo "============================================"
echo "NeuMan Preprocessing Pipeline"
echo "============================================"
echo "Video:    $VIDEO_ABS"
echo "Data dir: $DATA_ABS"
echo "SMPL-X:   $SMPLX_ABS"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Extract video frames (no GPU needed)
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/raw_720p" ]; then
    echo "[1/7] Extracting video frames..."
    docker run --rm \
        -v "$DATA_ABS":/data \
        -v "$(dirname $VIDEO_ABS)":/video:ro \
        neuman-step1-frames \
        --video "/video/$(basename $VIDEO_ABS)" \
        --save_to /data/raw_720p \
        --width 1280 --height 720 --every 10 --skip=0
else
    echo "[1/7] Skipping frame extraction (raw_720p exists)"
fi

# ------------------------------------------------------------------
# Step 2a: Generate person masks for raw frames
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/raw_masks" ]; then
    echo "[2/7] Generating person masks (raw frames)..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step2-detectron2 \
        masks /data/raw_720p /data/raw_masks
else
    echo "[2/7] Skipping raw masks (raw_masks exists)"
fi

# ------------------------------------------------------------------
# Step 3: COLMAP reconstruction
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/sparse" ]; then
    echo "[3/7] Running COLMAP sparse + dense reconstruction..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step3-colmap \
        /data/raw_720p /data/raw_masks /data
else
    echo "[3/7] Skipping COLMAP (output/sparse exists)"
fi

# ------------------------------------------------------------------
# Step 2b: Generate person masks for COLMAP-rectified images
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/segmentations" ]; then
    echo "[  ] Generating person masks (rectified images)..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step2-detectron2 \
        masks /data/output/images /data/output/segmentations
else
    echo "[  ] Skipping rectified masks (output/segmentations exists)"
fi

# ------------------------------------------------------------------
# Step 2c: DensePose on rectified images
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/densepose" ]; then
    echo "[  ] Running DensePose..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step2-detectron2 \
        densepose /data/output/images /data/output/densepose
else
    echo "[  ] Skipping DensePose (output/densepose exists)"
fi

# ------------------------------------------------------------------
# Step 4: 2D keypoint detection
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/keypoints" ]; then
    echo "[4/7] Running 2D keypoint detection..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step4-keypoints \
        /data/output/images /data/output/keypoints
else
    echo "[4/7] Skipping keypoints (output/keypoints exists)"
fi

# ------------------------------------------------------------------
# Step 5: Monocular depth estimation
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/mono_depth" ]; then
    echo "[5/7] Running monocular depth estimation..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step5-depth \
        /data/output/images /data/output/mono_depth
else
    echo "[5/7] Skipping depth (output/mono_depth exists)"
fi

# ------------------------------------------------------------------
# Step 6: SMPL parameter estimation
# ------------------------------------------------------------------
if [ ! -d "$DATA_ABS/output/smpl_pred" ]; then
    echo "[6/7] Running SMPL estimation (ROMP)..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        neuman-step6-smpl \
        /data/output/images /data/output/smpl_pred
else
    echo "[6/7] Skipping SMPL estimation (output/smpl_pred exists)"
fi

# ------------------------------------------------------------------
# Step 7: Scale alignment + SMPL optimization
# ------------------------------------------------------------------
if [ ! -f "$DATA_ABS/output/smpl_output_optimized.pkl" ]; then
    echo "[7/7] Running alignment and SMPL optimization..."
    docker run --rm --gpus all \
        -v "$DATA_ABS":/data \
        -v "$SMPLX_ABS":/neuman/data/smplx:ro \
        neuman-step7-neuman \
        all /data/output
else
    echo "[7/7] Skipping optimization (smpl_output_optimized.pkl exists)"
fi

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Output: $DATA_ABS/output/"
echo "============================================"
echo ""
echo "Contents:"
ls -la "$DATA_ABS/output/" 2>/dev/null || echo "(directory listing failed)"
echo ""
echo "Next steps:"
echo "  1. Check SMPL overlays in output/overlay/optimized/"
echo "  2. Use this output directory as the dataset path for HUGS training"
