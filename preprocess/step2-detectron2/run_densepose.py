#!/usr/bin/env python3
import argparse
import glob
import os
import pickle

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detectron2 DensePose inference.")
    parser.add_argument("input_dir", help="Directory containing input PNG/JPG images")
    parser.add_argument("output_dir", help="Directory to save DensePose output.pkl")
    parser.add_argument(
        "--config",
        default="/opt/conda/lib/python3.10/site-packages/densepose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml",
        help="DensePose config YAML path",
    )
    parser.add_argument(
        "--weights",
        default="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl",
        help="DensePose model weights",
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
    add_densepose_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    predictor = DefaultPredictor(cfg)
    extractor = DensePoseResultExtractor()

    images = sorted(
        glob.glob(os.path.join(args.input_dir, "*.png"))
        + glob.glob(os.path.join(args.input_dir, "*.jpg"))
    )
    print(f"Processing {len(images)} images for DensePose...")

    results = []
    for img_path in images:
        img = cv2.imread(img_path)
        with torch.no_grad():
            outputs = predictor(img)["instances"]
        result = extractor(outputs)
        results.append(
            {
                "file_name": os.path.basename(img_path),
                "scores": outputs.scores.cpu().numpy(),
                "pred_boxes_XYXY": outputs.pred_boxes.tensor.cpu().numpy(),
                "pred_densepose": result,
            }
        )

    output_path = os.path.join(args.output_dir, "output.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"DensePose results saved to {output_path}")


if __name__ == "__main__":
    main()
