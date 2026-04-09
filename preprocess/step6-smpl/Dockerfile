# Step 6: SMPL Body Model Estimation (ROMP)
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

# Install the upstream ROMP package (maintained version)
# simple_romp is the lightweight inference-only package
RUN pip install --no-cache-dir simple_romp

# Pre-download model weights
RUN python -c "from romp import ROMP; ROMP()(None)" 2>/dev/null; exit 0

COPY entrypoint.py /app/entrypoint.py
WORKDIR /app

# Usage:
#   docker run --rm --gpus all -v /path/to/data:/data step6-smpl \
#     /data/my_video/output/images /data/my_video/output/smpl_pred
ENTRYPOINT ["python", "/app/entrypoint.py"]
