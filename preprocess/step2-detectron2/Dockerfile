# Step 2: Instance Segmentation + DensePose (Detectron2)
# Handles original pipeline steps 2, 4, and 5
# GPU required
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh && conda clean -afy

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2 from source (supports PyTorch 2.x)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Pre-download model weights so they're baked into the image
RUN mkdir -p /models && \
    wget -q https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl \
      -O /models/mask_rcnn_X_101.pkl

COPY entrypoint.sh /app/entrypoint.sh
COPY run_masks.py /app/run_masks.py
COPY run_densepose.py /app/run_densepose.py
RUN chmod +x /app/entrypoint.sh /app/run_masks.py /app/run_densepose.py
WORKDIR /app

# Usage (see entrypoint.sh for subcommands):
#   docker run --rm --gpus all -v /path/to/data:/data step2-detectron2 \
#     masks /data/my_video/raw_720p /data/my_video/raw_masks
#
#   docker run --rm --gpus all -v /path/to/data:/data step2-detectron2 \
#     masks /data/my_video/output/images /data/my_video/output/segmentations
#
#   docker run --rm --gpus all -v /path/to/data:/data step2-detectron2 \
#     densepose /data/my_video/output/images /data/my_video/output/densepose
ENTRYPOINT ["/app/entrypoint.sh"]
