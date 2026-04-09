# Step 1: Extract video frames
# No GPU required — just OpenCV
FROM python:3.10-slim

RUN pip install --no-cache-dir opencv-python-headless Pillow imageio tqdm numpy

COPY save_video_frames.py /app/save_video_frames.py
WORKDIR /app

# Usage:
#   docker run --rm -v /path/to/data:/data step1-frames \
#     python save_video_frames.py \
#       --video /data/my_video.mp4 \
#       --save_to /data/my_video/raw_720p \
#       --width 1280 --height 720 --every 10 --skip=0
ENTRYPOINT ["python", "save_video_frames.py"]
