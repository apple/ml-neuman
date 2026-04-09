#!/bin/bash
set -e

IMAGE_DIR="$1"   # e.g., /data/my_video/raw_720p
MASK_DIR="$2"    # e.g., /data/my_video/raw_masks
WORK_DIR="$3"    # e.g., /data/my_video

if [ -z "$IMAGE_DIR" ] || [ -z "$MASK_DIR" ] || [ -z "$WORK_DIR" ]; then
    echo "Usage: entrypoint.sh <image_dir> <mask_dir> <work_dir>"
    echo "  image_dir: path to extracted 720p frames"
    echo "  mask_dir:  path to person masks from step 2"
    echo "  work_dir:  working directory (recon/ and output/ created here)"
    exit 1
fi

RECON_DIR="$WORK_DIR/recon"
OUTPUT_DIR="$WORK_DIR/output"

mkdir -p "$RECON_DIR/sparse" "$RECON_DIR/dense"

echo "=== Feature extraction ==="
colmap feature_extractor \
    --database_path "$RECON_DIR/db.db" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.mask_path "$MASK_DIR" \
    --SiftExtraction.estimate_affine_shape=true \
    --SiftExtraction.domain_size_pool=true \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1

echo "=== Feature matching ==="
colmap exhaustive_matcher \
    --database_path "$RECON_DIR/db.db" \
    --SiftMatching.guided_matching=true

echo "=== Sparse mapping ==="
colmap mapper \
    --database_path "$RECON_DIR/db.db" \
    --image_path "$IMAGE_DIR" \
    --output_path "$RECON_DIR/sparse"

# Check for good reconstruction (only one model)
if [ -d "$RECON_DIR/sparse/1" ]; then
    echo "WARNING: COLMAP produced multiple models. Reconstruction may be poor."
    echo "Continuing with model 0, but results may be unreliable."
fi

echo "=== Image undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$RECON_DIR/sparse/0/" \
    --output_path "$RECON_DIR/dense"

echo "=== Patch match stereo ==="
colmap patch_match_stereo \
    --workspace_path "$RECON_DIR/dense"

echo "=== Model export ==="
colmap model_converter \
    --input_path "$RECON_DIR/dense/sparse/" \
    --output_path "$RECON_DIR/dense/sparse" \
    --output_type=TXT

echo "=== Copying to output ==="
mkdir -p "$OUTPUT_DIR"
cp -r "$RECON_DIR/dense/images" "$OUTPUT_DIR/images"
cp -r "$RECON_DIR/dense/stereo/depth_maps" "$OUTPUT_DIR/depth_maps"
cp -r "$RECON_DIR/dense/sparse" "$OUTPUT_DIR/sparse"

echo "=== COLMAP complete. Output at $OUTPUT_DIR ==="
