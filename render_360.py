#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render 360 views of a Da-posed human.
Render 360 views of a posed human.

Examples:
python render_360.py --scene_dir ./data/seattle2 --use_cuda=no --white_bkg=yes --rays_per_batch=2048 --trajectory_resolution=40 --weights_path ./out/seattle2_rotate/checkpoint.pth.tar --render_h=72 --render_w=128 --mode canonical_360 --can_posenc rotate
python render_360.py --scene_dir ./data/seattle2 --use_cuda=no --white_bkg=yes --rays_per_batch=2048 --trajectory_resolution=40 --weights_path ./out/seattle2_rotate/checkpoint.pth.tar --render_h=72 --render_w=128 --mode posed_360 --can_posenc rotate
'''
import os
import argparse

import imageio
import torch
import numpy as np

from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera
from models import human_nerf
from utils import render_utils, utils
from data_io import neuman_helper
from options import options
from utils.constant import CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST


def main_canonical_360(opt):
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.render_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale
    )
    if opt.geo_threshold < 0:
        can_bones = []
        bones = []
        for i in range(len(scene.captures)):
            bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
            can_bones.append(np.linalg.norm(scene.smpls[i]['static_joints_3d'][3] - scene.smpls[i]['static_joints_3d'][0]))
        opt.geo_threshold = np.mean(bones)
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    center, up = utils.smpl_verts_to_center_and_up(scene.static_vert[0])
    render_poses = render_utils.default_360_path(center, up, CANONICAL_CAMERA_DIST, opt.trajectory_resolution)

    for i, rp in enumerate(render_poses):
        can_cap = ResizedPinholeCapture(
            PinholeCamera(
                scene.captures[0].pinhole_cam.width,
                scene.captures[0].pinhole_cam.height,
                CANONICAL_ZOOM_FACTOR * scene.captures[0].pinhole_cam.width,
                CANONICAL_ZOOM_FACTOR * scene.captures[0].pinhole_cam.width,
                scene.captures[0].pinhole_cam.width / 2.0,
                scene.captures[0].pinhole_cam.height / 2.0,
            ),
            rp,
            tgt_size=scene.captures[0].pinhole_cam.shape
        )
        out = render_utils.render_smpl_nerf(
            net,
            can_cap,
            scene.static_vert[0],
            scene.faces,
            Ts=None,
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            render_can=True,
            return_mask=False,
            return_depth=False,
            interval_comp=opt.geo_threshold / np.mean(can_bones)
        )
        save_path = os.path.join('./demo', f'canonical_360/{os.path.basename(opt.scene_dir)}', f'out_{str(i).zfill(4)}.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        imageio.imsave(save_path, out)
        print(f'image saved: {save_path}')


def main_posed_360(opt):
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.render_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale,
        smpl_type='optimized'
    )
    if opt.geo_threshold < 0:
        bones = []
        for i in range(len(scene.captures)):
            bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
        opt.geo_threshold = np.mean(bones)
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    cap_id = 0
    center, up = utils.smpl_verts_to_center_and_up(scene.verts[cap_id])
    dist = opt.geo_threshold * 36 # camera distance depends on the human size
    render_poses = render_utils.default_360_path(center, up, dist, opt.trajectory_resolution)

    for i, rp in enumerate(render_poses):
        can_cap = ResizedPinholeCapture(
            scene.captures[0].pinhole_cam,
            rp,
            tgt_size=scene.captures[0].size
        )
        out = render_utils.render_smpl_nerf(
            net,
            can_cap,
            scene.verts[cap_id],
            scene.faces,
            scene.Ts[cap_id],
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            white_bkg=opt.white_bkg,
            render_can=False,
            geo_threshold=opt.geo_threshold
        )
        save_path = os.path.join('./demo', f'posed_360/{os.path.basename(opt.scene_dir)}', f'out_{str(i).zfill(4)}.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        imageio.imsave(save_path, out)
        print(f'image saved: {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--image_dir', required=False, type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--mode', required=True, choices=['canonical_360', 'posed_360'], type=str, help='rendering mode')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')

    opt = parser.parse_args()
    assert opt.geo_threshold == -1, 'please use auto geo_threshold'
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    options.print_opt(opt)
    if opt.mode == 'canonical_360':
        main_canonical_360(opt)
    elif opt.mode == 'posed_360':
        main_posed_360(opt)
