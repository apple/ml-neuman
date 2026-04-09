# Step 3: COLMAP Structure-from-Motion + Dense Reconstruction
# GPU required (for patch_match_stereo)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential wget ca-certificates \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev \
    libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libcgal-qt5-dev \
    libflann-dev libsqlite3-dev libmetis-dev libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Ceres Solver
RUN git clone https://github.com/ceres-solver/ceres-solver.git --branch 2.1.0 --depth 1 && \
    cd ceres-solver && mkdir build && cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -GNinja && \
    ninja -j$(nproc) && ninja install && \
    cd / && rm -rf ceres-solver

# Build COLMAP with broad GPU arch support
RUN git clone https://github.com/colmap/colmap.git --branch 3.9.1 --depth 1 && \
    cd colmap && mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" && \
    ninja -j$(nproc) && ninja install && \
    cd / && rm -rf colmap

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /app

# Usage:
#   docker run --rm --gpus all -v /path/to/data:/data step3-colmap \
#     /data/my_video/raw_720p /data/my_video/raw_masks /data/my_video
ENTRYPOINT ["/app/entrypoint.sh"]
