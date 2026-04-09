"""
Step 5: Monocular Depth Estimation using Depth Anything v2

Saves depth maps as 16-bit PNGs (NeuMan reads them as depth = pixel_value / 10000.0).
"""

import sys
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    if len(sys.argv) != 3:
        print("Usage: entrypoint.py <input_images_dir> <output_depth_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from transformers import pipeline

    # Use Depth Anything v2 Large for best quality
    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device="cuda:0",
        torch_dtype=torch.float16,
    )

    images = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
    )
    print(f"Processing {len(images)} images...")

    for img_path in tqdm(images, desc="Depth estimation"):
        img = Image.open(img_path).convert("RGB")

        result = depth_pipe(img)
        depth = np.array(result["depth"], dtype=np.float32)

        # Normalize to a reasonable metric-like range and convert to 16-bit PNG.
        # NeuMan reads mono depth as relative (used for scale alignment, not
        # absolute metric depth), so the exact scale doesn't matter much —
        # it gets calibrated against COLMAP in step 9.
        # We scale so values fit well in uint16 range with the /10000 convention.
        depth_norm = depth / (depth.max() + 1e-8)  # 0-1
        depth_uint16 = (depth_norm * 10000).astype(np.uint16)

        fname = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        Image.fromarray(depth_uint16).save(os.path.join(output_dir, fname))

    print(f"Depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
