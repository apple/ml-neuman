"""
Step 6: SMPL estimation using ROMP (simple_romp package)

Saves per-image SMPL predictions in the format NeuMan expects.
"""

import sys
import os
import glob
import numpy as np
from tqdm import tqdm


def main():
    if len(sys.argv) != 3:
        print("Usage: entrypoint.py <input_images_dir> <output_smpl_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from romp import ROMP

    model = ROMP(mode='image')

    images = sorted(
        glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpg"))
    )
    print(f"Processing {len(images)} images for SMPL estimation...")

    all_results = {}
    for img_path in tqdm(images, desc="ROMP"):
        import cv2
        img = cv2.imread(img_path)
        outputs = model(img)

        fname = os.path.basename(img_path)
        if outputs is not None:
            all_results[fname] = {
                'smpl_thetas': outputs['smpl_thetas'].cpu().numpy(),
                'smpl_betas': outputs['smpl_betas'].cpu().numpy(),
                'cam_trans': outputs['cam_trans'].cpu().numpy(),
                'center_conf': outputs['center_conf'].cpu().numpy()
                    if 'center_conf' in outputs else None,
                'verts': outputs['verts'].cpu().numpy()
                    if 'verts' in outputs else None,
            }
        else:
            all_results[fname] = None

    # Save as pickle (NeuMan's export_alignment.py reads this format)
    import pickle
    output_path = os.path.join(output_dir, 'results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Also save per-image npz files for compatibility
    for fname, result in all_results.items():
        if result is not None:
            npz_path = os.path.join(output_dir, fname + '.npz')
            np.savez(npz_path, **result)

    print(f"SMPL predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
