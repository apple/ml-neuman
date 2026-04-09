#!/usr/bin/env python3
import argparse
import glob
import os

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detectron2 instance segmentation masks.")
    parser.add_argument("input_dir", help="Directory containing input PNG/JPG images")
    parser.add_argument("output_dir", help="Directory to save output masks")
    parser.add_argument(
        "--weights",
        default="/models/mask_rcnn_X_101.pkl",
        help="Path to model weights",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="ROI head score threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    predictor = DefaultPredictor(cfg)

    images = sorted(
        glob.glob(os.path.join(args.input_dir, "*.png"))
        + glob.glob(os.path.join(args.input_dir, "*.jpg"))
    )
    print(f"Processing {len(images)} images...")

    for img_path in images:
        img = cv2.imread(img_path)
        outputs = predictor(img)

        # 0 where person detected, 255 elsewhere (COLMAP convention)
        instances = outputs["instances"].to("cpu")
        person_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for i in range(len(instances)):
            if instances.pred_classes[i] == 0:  # person class
                mask = instances.pred_masks[i].numpy()
                person_mask = np.logical_or(person_mask, mask)

        # COLMAP masks out black regions, so invert the mask.
        out_mask = np.where(person_mask, 0, 255).astype(np.uint8)

        fname = os.path.basename(img_path).replace(".jpg", ".png")
        cv2.imwrite(os.path.join(args.output_dir, fname + ".png"), out_mask)

    print(f"Masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
