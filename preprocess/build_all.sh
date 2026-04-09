#!/bin/bash
# =============================================================================
# Build all 7 Docker images for the NeuMan preprocessing pipeline
#
# Usage:
#   ./build_all.sh
#
# Build times (approximate, depends on network + hardware):
#   step1-frames:      ~30 seconds  (tiny, no GPU libs)
#   step2-detectron2:  ~10 minutes  (PyTorch + Detectron2 from source)
#   step3-colmap:      ~15 minutes  (compiles COLMAP + Ceres from source)
#   step4-keypoints:   ~8 minutes   (PyTorch + mmpose)
#   step5-depth:       ~8 minutes   (PyTorch + HuggingFace model download)
#   step6-smpl:        ~5 minutes   (PyTorch + simple_romp)
#   step7-neuman:      ~15 minutes  (PyTorch + PyTorch3D from source)
#
# Total: ~60 minutes first build, cached rebuilds much faster
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building all pipeline images..."
echo ""

build_step() {
    local name="$1"
    local dir="$2"
    echo "============================================"
    echo "Building: $name"
    echo "============================================"
    docker build -t "$name" "$SCRIPT_DIR/$dir"
    echo ""
}

build_step "neuman-step1-frames"     "step1-frames"
build_step "neuman-step2-detectron2" "step2-detectron2"
build_step "neuman-step3-colmap"     "step3-colmap"
build_step "neuman-step4-keypoints"  "step4-keypoints"
build_step "neuman-step5-depth"      "step5-depth"
build_step "neuman-step6-smpl"       "step6-smpl"
build_step "neuman-step7-neuman"     "step7-neuman"

echo "============================================"
echo "All images built successfully!"
echo "============================================"
echo ""
docker images | grep neuman-step
