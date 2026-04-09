# =============================================================================
# Updated NeuMan Preprocessing Dockerfile
# Original: CUDA 10.2 / Ubuntu 18.04 / PyTorch 1.8-1.9 / Python 3.7-3.8
# Updated:  CUDA 11.8 / Ubuntu 22.04 / PyTorch 2.1   / Python 3.10
#
# Supports: RTX 30xx, 40xx, 50xx, A100, H100, etc.
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# ---- System packages (COLMAP build deps + general tools) --------------------
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    vim \
    wget \
    curl \
    unzip \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libflann-dev \
    libsqlite3-dev \
    libmetis-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Build Ceres Solver (required by COLMAP) --------------------------------
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 2.1.0 --depth 1 && \
    cd ceres-solver && \
    mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -GNinja && \
    ninja -j$(nproc) && \
    ninja install && \
    cd / && rm -rf ceres-solver

# ---- Build COLMAP (3.9.1 — latest stable with CUDA support) -----------------
RUN git clone https://github.com/colmap/colmap.git --branch 3.9.1 --depth 1 && \
    cd colmap && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" && \
    ninja -j$(nproc) && \
    ninja install && \
    cd / && rm -rf colmap

# ---- Install Miniconda ------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda init bash

# ---- Clone NeuMan and dependency repos --------------------------------------
#
# NOTE: We keep the original author's forks (jiangwei221) because they contain
# NeuMan-specific patches. The specific commits are pinned for reproducibility.
# If you encounter build issues with these old forks, see the comments below
# for upstream alternatives.
#
RUN git clone https://github.com/apple/ml-neuman.git && mv ml-neuman neuman

RUN cd /neuman/preprocess && \
    git clone --recurse-submodules https://github.com/compphoto/BoostingMonocularDepth.git && \
    cd BoostingMonocularDepth && \
    git checkout ecedd0c0cf5e1807cdab1c5154351a97168e710d

RUN cd /neuman/preprocess && \
    git clone --recurse-submodules https://github.com/jiangwei221/ROMP.git && \
    cd ROMP && \
    git checkout f1aaf0c1d90435bbeabe39cf04b15e12906c6111

# For Detectron2, we use the upstream repo which officially supports PyTorch 2.x.
# The original NeuMan fork (jiangwei221/detectron2) was based on an older version
# and won't compile against PyTorch 2.x. The demo scripts and model weights
# remain compatible.
RUN cd /neuman/preprocess && \
    git clone https://github.com/facebookresearch/detectron2.git

# For mmpose, we keep the original fork since the pipeline uses the old
# bottom_up_img_demo.py API which was removed in mmpose 1.x.
# We'll patch any compilation issues below.
RUN cd /neuman/preprocess && \
    git clone --recurse-submodules https://github.com/jiangwei221/mmpose.git && \
    cd mmpose && \
    git checkout 8b788f93200ce6485e885da0c736f114e4de8eaf

# =============================================================================
# ROMP environment (also used for Detectron2, DensePose, depth estimation)
# =============================================================================
# This is the main workhorse environment — handles steps 1,2,4,5,7,8
RUN . /root/.bashrc && \
    conda create -n ROMP python=3.10 -y && \
    conda activate ROMP && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install numpy==1.26.4 scipy cython

# Install Detectron2 from source (supports PyTorch 2.x)
RUN . /root/.bashrc && \
    conda activate ROMP && \
    cd /neuman/preprocess/detectron2 && \
    pip install -e . && \
    pip install setuptools==69.5.1

# Install ROMP dependencies
# NOTE: The pinned ROMP fork is old. Some of its requirements.txt pins may
# conflict with PyTorch 2.x. We install core deps manually and then try
# the requirements file, ignoring version conflicts.
RUN . /root/.bashrc && \
    conda activate ROMP && \
    cd /neuman/preprocess/ROMP && \
    pip install opencv-python-headless Pillow scikit-image tqdm joblib \
        h5py yacs trimesh pycocotools && \
    pip install av && \
    pip install --no-deps -r requirements.txt 2>/dev/null; exit 0

# Install PyTorch3D (needed by ROMP and neuman_env scripts)
# Build from source for CUDA 11.8 + PyTorch 2.1 compatibility
RUN . /root/.bashrc && \
    conda activate ROMP && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install BoostingMonocularDepth deps
RUN . /root/.bashrc && \
    conda activate ROMP && \
    pip install timm

# =============================================================================
# open-mmlab environment (for mmpose 2D keypoint detection — step 6)
# =============================================================================
# The old mmpose fork requires mmcv-full (now called mmcv) and the old API.
# We use PyTorch 2.1 but install the legacy mmcv build.
RUN . /root/.bashrc && \
    conda create -n open-mmlab python=3.10 -y && \
    conda activate open-mmlab && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install mmcv (the successor to mmcv-full)
# For the OLD mmpose (0.x) we need mmcv-full <= 1.7.x
# The prebuilt wheels for cu118+torch2.1 may not exist for mmcv-full,
# so we build from source.
RUN . /root/.bashrc && \
    conda activate open-mmlab && \
    pip install openmim && \
    mim install mmcv==2.1.0 && \
    pip install mmdet mmengine

# Install the old mmpose fork
# This may require patching imports from mmcv to work with mmcv 2.x.
# If you get import errors, the alternative is to use mmpose 1.x with
# its new inference API (see ALTERNATIVE section below).
RUN . /root/.bashrc && \
    conda activate open-mmlab && \
    cd /neuman/preprocess/mmpose && \
    pip install -r requirements.txt 2>/dev/null; \
    pip install -v -e . 2>/dev/null; exit 0

# =============================================================================
# NeuMan environment (for scale alignment and SMPL optimization — steps 9,10)
# =============================================================================
RUN . /root/.bashrc && \
    conda create -n neuman_env python=3.10 -y && \
    conda activate neuman_env && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    pip install opencv-python-headless joblib open3d imageio \
        tensorboardX chumpy lpips scikit-image ipython matplotlib \
        numpy==1.26.4 scipy trimesh

# Install igl (libigl Python bindings)
RUN . /root/.bashrc && \
    conda activate neuman_env && \
    pip install libigl

# =============================================================================
# ALTERNATIVE: If the old mmpose fork fails to build, replace step 6 with
# mmpose 1.x. You'd need to:
#
# 1. Install modern mmpose:
#      conda activate open-mmlab
#      mim install mmpose
#
# 2. Update gen_run.py step 6 command to use the new inference API:
#      python demo/inferencer_demo.py \
#        <input_images> \
#        --pose2d configs/body_2d_keypoint/associative_embedding/coco/ae_higherhrnet-w48_8xb24-300e_coco-512x512.py \
#        --pose2d-weights https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/associative_embedding/coco/ae_higherhrnet-w48_8xb24-300e_coco-512x512-09b0b857_20230908.pth \
#        --out-dir <output_dir>
#
# The output format may differ slightly — you may need to adapt
# export_alignment.py and optimize_smpl.py to read the new keypoint format.
# =============================================================================

WORKDIR /neuman/preprocess