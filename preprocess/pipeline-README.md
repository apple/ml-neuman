# NeuMan Preprocessing Pipeline (Modernized)

Containerized, step-by-step preprocessing pipeline for preparing videos for [NeuMan](https://github.com/apple/ml-neuman) and [HUGS](https://github.com/apple/ml-hugs) training.

Replaces the original monolithic Dockerfile (CUDA 10.2 / Ubuntu 18.04 / PyTorch 1.8) with 7 independent containers running on modern hardware (CUDA 11.8 / Ubuntu 22.04 / PyTorch 2.1).

## Requirements

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (RTX 30xx, 40xx, 50xx, A100, H100, etc.)
- SMPL-X model files (register at https://smpl-x.is.tue.mpg.de/)

## Quick Start

```bash
# 1. Build all images (~60 min first time)
chmod +x build_all.sh
./build_all.sh

# 2. Run the full pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh /path/to/video.mp4 /path/to/working_dir /path/to/smplx
```

## Pipeline Steps

| Step | Container | GPU | What it does | Replaces |
|------|-----------|-----|-------------|----------|
| 1 | `step1-frames` | No | Extract 720p frames from video | `save_video_frames.py` |
| 2 | `step2-detectron2` | Yes | Instance segmentation masks + DensePose | Detectron2 fork (steps 2,4,5) |
| 3 | `step3-colmap` | Yes | SfM + dense stereo reconstruction | COLMAP (step 3) |
| 4 | `step4-keypoints` | Yes | 2D body keypoint detection | mmpose fork (step 6) |
| 5 | `step5-depth` | Yes | Monocular depth estimation | BoostingMonocularDepth (step 7) |
| 6 | `step6-smpl` | Yes | SMPL body model estimation | ROMP fork (step 8) |
| 7 | `step7-neuman` | Yes | Scale alignment + SMPL silhouette optimization | NeuMan scripts (steps 9,10) |

## Key Upgrades Over the Original

| Component | Original (2022) | Updated |
|-----------|----------------|---------|
| Base OS | Ubuntu 18.04 (EOL) | Ubuntu 22.04 |
| CUDA | 10.2 (no RTX 30xx+ support) | 11.8 (all modern GPUs) |
| Python | 3.7 / 3.8 | 3.10 |
| PyTorch | 1.8 / 1.9 | 2.1.2 |
| Detectron2 | Forked, pinned 2022 commit | Upstream, PyTorch 2.x compatible |
| mmpose | Forked v0.25 (old API) | v1.x with MMPoseInferencer |
| Depth | BoostingMonocularDepth (dead URLs) | Depth Anything v2 (HuggingFace) |
| ROMP | Forked, pinned commit | `simple_romp` pip package |
| COLMAP | Pinned old commit | 3.9.1 stable release |
| Conda envs | 3 separate environments | 1 per container (isolated) |

## Running Steps Individually

Each container can be run independently. This is useful for debugging or re-running a single failed step:

```bash
# Just extract frames
docker run --rm -v /data:/data neuman-step1-frames \
    --video /data/my_video.mp4 --save_to /data/raw_720p \
    --width 1280 --height 720 --every 10

# Just run DensePose
docker run --rm --gpus all -v /data:/data neuman-step2-detectron2 \
    densepose /data/output/images /data/output/densepose

# Just run SMPL optimization
docker run --rm --gpus all \
    -v /data:/data -v /path/to/smplx:/neuman/data/smplx:ro \
    neuman-step7-neuman optimize /data/output
```

## Resumability

The orchestration script (`run_pipeline.sh`) checks for existing output directories before running each step. If a step has already been completed, it's skipped automatically. To force re-running a step, delete its output directory.

## Output Structure

After the pipeline completes, `<data_dir>/output/` will contain:

```
output/
├── images/           # COLMAP-undistorted frames
├── sparse/           # COLMAP camera params (cameras.txt, images.txt, points3D.txt)
├── depth_maps/       # COLMAP stereo depth maps
├── segmentations/    # Per-frame person masks (rectified)
├── densepose/        # DensePose output (output.pkl)
├── keypoints/        # Per-frame 2D keypoints (.npy)
├── mono_depth/       # Monocular depth maps (16-bit PNG)
├── smpl_pred/        # Raw ROMP SMPL predictions
├── alignments.npy    # Scene-SMPL scale alignment
├── smpl_output_optimized.pkl  # Final optimized SMPL parameters
└── overlay/          # Visual check overlays
```

This directory is directly usable as a NeuMan / HUGS dataset path.

## Troubleshooting

**COLMAP produces multiple models (warning in step 3):**
This usually means the video has too few frames, insufficient texture, or too much motion blur. Try adjusting `--every` in step 1 to sample more frames, or use a different segment of the video.

**DensePose entrypoint fails with import errors:**
The DensePose project within Detectron2 can have version-specific import paths. If `from densepose import add_densepose_config` fails, try `from detectron2.projects.densepose import add_densepose_config` or install densepose separately: `pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose`.

**Depth maps look wrong:**
Depth Anything v2 produces inverse depth (closer = higher value), which is the same convention as BoostingMonocularDepth. If NeuMan expects the opposite, flip with: `depth = depth.max() - depth`.

**SMPL alignment fails:**
Check that `smpl_pred/` contains valid predictions for most frames. ROMP sometimes fails on frames with heavy occlusion or unusual poses. Also verify that SMPL-X model files are correctly mounted at `/neuman/data/smplx/`.
