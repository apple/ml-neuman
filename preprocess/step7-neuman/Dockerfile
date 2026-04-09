# Step 7: Scale Alignment + SMPL Optimization (NeuMan custom scripts)
# Handles original pipeline steps 9 and 10
# GPU required (pytorch3d rendering for silhouette optimization)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh && conda clean -afy

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# PyTorch3D from source (needed for differentiable rendering in SMPL optimization)
# devel image required for CUDA compiler
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN pip install --no-cache-dir \
    opencv-python-headless joblib open3d imageio numpy==1.26.4 scipy \
    tensorboardX chumpy lpips scikit-image ipython matplotlib trimesh libigl

# Clone NeuMan repo (we need the full codebase for data_io, utils, models, etc.)
RUN git clone https://github.com/apple/ml-neuman.git /neuman

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /neuman/preprocess

# Usage:
#   docker run --rm --gpus all \
#     -v /path/to/data:/data \
#     -v /path/to/smplx:/neuman/data/smplx \
#     step7-neuman \
#     align /data/my_video/output
#
#   docker run --rm --gpus all \
#     -v /path/to/data:/data \
#     -v /path/to/smplx:/neuman/data/smplx \
#     step7-neuman \
#     optimize /data/my_video/output
#
#   docker run --rm --gpus all \
#     -v /path/to/data:/data \
#     -v /path/to/smplx:/neuman/data/smplx \
#     step7-neuman \
#     all /data/my_video/output
ENTRYPOINT ["/app/entrypoint.sh"]
