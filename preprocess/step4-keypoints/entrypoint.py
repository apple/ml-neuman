"""
Step 4: 2D Keypoint Detection using mmpose 1.x

Runs bottom-up pose estimation and saves per-image keypoint .npy files
in the format expected by NeuMan: (N_people, 17, 3) where last dim is [x, y, score].
"""

import sys
import os
import glob
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: entrypoint.py <input_images_dir> <output_keypoints_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    from mmpose.apis import MMPoseInferencer

    # Use HigherHRNet-w48 bottom-up model (closest to original pipeline)
    # Falls back to RTMPose if HigherHRNet config unavailable
    try:
        inferencer = MMPoseInferencer(
            pose2d='td-hm_HigherHRNet-w48_8xb24-300e_coco-512x512',
            device='cuda:0'
        )
        print("Using HigherHRNet-w48 (bottom-up)")
    except Exception:
        print("HigherHRNet not available, falling back to RTMPose (top-down)")
        inferencer = MMPoseInferencer(pose2d='human', device='cuda:0')

    images = sorted(
        glob.glob(os.path.join(input_dir, '*.png')) +
        glob.glob(os.path.join(input_dir, '*.jpg'))
    )
    print(f"Processing {len(images)} images...")

    for img_path in images:
        results = list(inferencer(img_path, show=False))

        if len(results) > 0 and len(results[0]['predictions']) > 0:
            preds = results[0]['predictions'][0]
            # Each prediction has 'keypoints' (N,K,2) and 'keypoint_scores' (N,K)
            all_kps = []
            for pred in preds:
                kp = np.array(pred['keypoints'])          # (17, 2)
                scores = np.array(pred['keypoint_scores']) # (17,)
                kp_with_score = np.concatenate(
                    [kp, scores[:, None]], axis=-1
                )  # (17, 3)
                all_kps.append(kp_with_score)
            keypoints = np.array(all_kps)  # (N_people, 17, 3)
        else:
            keypoints = np.zeros((0, 17, 3))

        # Save as .npy with same naming convention as original pipeline
        fname = os.path.basename(img_path) + '.npy'
        np.save(os.path.join(output_dir, fname), keypoints)

    print(f"Keypoints saved to {output_dir}")


if __name__ == "__main__":
    main()
