#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Preprocess
'''

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()
    assert os.path.isfile(opt.video)

    steps = 10
    video_name = os.path.splitext(os.path.basename(opt.video))[0]
    code_dir = os.path.dirname(os.path.realpath(__file__))
    video_dir = os.path.dirname(os.path.abspath(opt.video))
    print(code_dir)

    commands = []
    commands.append('#!/bin/bash')
    # Extract video frames
    commands.append(f'echo ========================================')
    commands.append(f'echo 1/{steps}: Extract frames')
    commands.append(f'echo ========================================')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/raw_720p')):
        commands.append('conda activate ROMP')
        commands.append(f'python save_video_frames.py --video {opt.video} --save_to {os.path.join(video_dir, video_name, "raw_720p")}  --width 1280 --height 720 --every 10 --skip=0')
        commands.append('conda deactivate')

    # Generate masks
    commands.append(f'echo ========================================')
    commands.append(f'echo 2/{steps}: Masks')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "detectron2/demo")}')
    if not os.path.isfile(os.path.join(code_dir, 'detectron2/demo/model_final_2d9806.pkl')):
        commands.append('wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/raw_masks')):
        commands.append('conda activate ROMP')
        commands.append(f'python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input {os.path.join(video_dir, f"{video_name}/raw_720p/*.png")} --output {os.path.join(video_dir, f"{video_name}/raw_masks")}  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Colmap
    commands.append(f'echo ========================================')
    commands.append(f'echo 3/{steps}: Sparse scene reconstrution')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(video_dir, video_name)}')
    if not os.path.isdir(os.path.join(video_dir, video_name, 'output/sparse')):
        commands.append('mkdir recon')
        commands.append('colmap feature_extractor --database_path ./recon/db.db --image_path ./raw_720p --ImageReader.mask_path ./raw_masks --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pool=true --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1')
        commands.append('colmap exhaustive_matcher --database_path ./recon/db.db --SiftMatching.guided_matching=true')
        # commands.append('')
        commands.append('mkdir -p ./recon/sparse')
        commands.append('colmap mapper --database_path ./recon/db.db --image_path ./raw_720p --output_path ./recon/sparse')
        commands.append('if [ -d "./recon/sparse/1" ]; then echo "Bad reconstruction"; exit 1; else echo "Ok"; fi')
        commands.append('mkdir -p ./recon/dense')
        commands.append('colmap image_undistorter --image_path raw_720p --input_path ./recon/sparse/0/ --output_path ./recon/dense')
        commands.append('colmap patch_match_stereo --workspace_path ./recon/dense')
        commands.append('colmap model_converter --input_path ./recon/dense/sparse/ --output_path ./recon/dense/sparse --output_type=TXT')
        commands.append('mkdir ./output')
        commands.append('cp -r ./recon/dense/images ./output/images')
        commands.append('cp -r ./recon/dense/stereo/depth_maps ./output/depth_maps')
        commands.append('cp -r ./recon/dense/sparse ./output/sparse')
    commands.append(f'cd {code_dir}')

    # Generate masks for rectified images
    commands.append(f'echo ========================================')
    commands.append(f'echo 4/{steps}: Masks for rectified images')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "detectron2/demo")}')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/segmentations')):
        commands.append('conda activate ROMP')
        commands.append(f'python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input {os.path.join(video_dir, f"{video_name}/output/images/*.png")} --output {os.path.join(video_dir, f"{video_name}/output/segmentations")}  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Run DensePose
    commands.append(f'echo ========================================')
    commands.append(f'echo 5/{steps}: DensePose')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "detectron2/projects/DensePose")}')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/densepose')):
        commands.append('conda activate ROMP')
        commands.append(f'python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl {os.path.join(video_dir, f"{video_name}/output/images")} {os.path.join(video_dir, f"{video_name}/output/densepose")}  --output {os.path.join(video_dir, f"{video_name}/output/densepose/output.pkl")} -v')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Run 2D keypoints detector(mmpose)
    commands.append(f'echo ========================================')
    commands.append(f'echo 6/{steps}: 2D keypoints')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "mmpose")}')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/keypoints')):
        commands.append('conda activate open-mmlab')
        commands.append(f'python demo/bottom_up_img_demo.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth --img-path {os.path.join(video_dir, f"{video_name}/output/images")} --out-img-root {os.path.join(video_dir, f"{video_name}/output/keypoints")} --kpt-thr=0.3 --pose-nms-thr=0.9')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Monocular depth estimation
    commands.append(f'echo ========================================')
    commands.append(f'echo 7/{steps}: Monocular depth')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "BoostingMonocularDepth")}')
    if not os.path.exists(os.path.join(code_dir, 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel')):
        os.makedirs(os.path.join(code_dir, 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel'))
    if not os.path.isfile(os.path.join(code_dir, 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth')):
        commands.append(f'wget https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth -O {os.path.join(code_dir, "BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth")}')
    if not os.path.isfile(os.path.join(code_dir, 'BoostingMonocularDepth/res101.pth')):
        commands.append(f'wget https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download -O res101.pth')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/mono_depth')):
        commands.append('conda activate ROMP')
        commands.append(f'python run.py --Final --data_dir {os.path.join(video_dir, f"{video_name}/output/images")} --output_dir {os.path.join(video_dir, f"{video_name}/output/mono_depth")} --depthNet 2')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # SMPL parameters estimation
    commands.append(f'echo ========================================')
    commands.append(f'echo 8/{steps}: SMPL parameters')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(code_dir, "ROMP")}')
    if not os.path.exists(os.path.join(code_dir, 'ROMP/model_data')):
        commands.append('wget https://github.com/jiangwei221/ROMP/releases/download/v1.1/model_data.zip')
        commands.append('unzip model_data.zip')
    if not os.path.exists(os.path.join(code_dir, 'ROMP/model_data')):
        commands.append('wget https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models_try.zip')
        commands.append('unzip trained_models_try.zip')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/smpl_pred')):
        commands.append('conda activate ROMP')
        commands.append(f'python -m romp.predict.image --inputs {os.path.join(video_dir, f"{video_name}/output/images")} --output_dir {os.path.join(video_dir, f"{video_name}/output/smpl_pred")}')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Solve scale ambiguity
    commands.append(f'echo ========================================')
    commands.append(f'echo 9/{steps}: Solve scale ambiguity')
    commands.append(f'echo ========================================')
    commands.append(f'cd {code_dir}')
    if not os.path.isfile(os.path.join(video_dir, f'{video_name}/output/alignments.npy')):
        commands.append('conda activate neuman_env')
        commands.append(f'python export_alignment.py --scene_dir {os.path.join(video_dir, f"{video_name}/output/sparse")} --images_dir {os.path.join(video_dir, f"{video_name}/output/images")} --raw_smpl {os.path.join(video_dir, f"{video_name}/output/smpl_pred")} --smpl_estimator="romp"')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # Optimize SMPL using silhouette
    commands.append(f'echo ========================================')
    commands.append(f'echo 10/{steps}: Optimize SMPL using silhouette')
    commands.append(f'echo ========================================')
    commands.append(f'cd {code_dir}')
    if not os.path.isfile(os.path.join(video_dir, f'{video_name}/output/smpl_output_optimized.pkl')):
        commands.append('conda activate neuman_env')
        commands.append(f'python optimize_smpl.py --scene_dir {os.path.join(video_dir, f"{video_name}/output")}')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    print(*commands, sep='\n')
    with open("run.sh", "w") as outfile:
        outfile.write("\n".join(commands))


if __name__ == "__main__":
    main()
