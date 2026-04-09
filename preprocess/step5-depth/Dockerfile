# Step 5: Monocular Depth Estimation (Depth Anything v2)
# Replaces BoostingMonocularDepth with a modern, better-quality model
# whose weights are readily available via HuggingFace.
# GPU required
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh && conda clean -afy

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    transformers huggingface_hub Pillow numpy opencv-python-headless tqdm

# Pre-download the model weights
RUN python -c "from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Large-hf', device='cpu')"

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Usage:
#   docker run --rm --gpus all -v /path/to/data:/data step5-depth \
#     /data/my_video/output/images /data/my_video/output/mono_depth
ENTRYPOINT ["python", "/app/entrypoint.py"]
