#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
The driving motions used in this project was obtained from http://mocap.cs.sfu.ca. The database was created with funding from NUS AcRF R-252-000-429-133 and SFU Presidents Research Start-up Grant.

We manually align the human to the scene using Blender, and hard code the alignments.
We manually create the novel camera paths.

Example:
python render_reposing.py --scene_dir ./data/bike --use_cuda=no --rays_per_batch=2048  --weights_path ./out/bike_rotate/checkpoint.pth.tar  --samples_per_ray=128 --motion_name=jumpandroll --bkg_range_scale=3 --can_posenc=rotate --render_h=72 --render_w=128
'''

import os
import argparse
import copy

import imageio
import torch
import numpy as np

from geometry import transformations
from models import human_nerf
from utils import render_utils, utils, ray_utils
from data_io import neuman_helper
from options import options
from models.smpl import SMPL


def read_novel_caps(opt, num_caps, scene):
    novel_caps = []
    if os.path.basename(opt.scene_dir) == 'bike' and opt.motion_name == 'jumpandroll':
        start_id = 25
        interval = 0.05
        for i in range(num_caps):
            temp = copy.deepcopy(scene.captures[start_id])
            temp.cam_pose.camera_center_in_world += interval * i * temp.cam_pose.right
            novel_caps.append(temp)
    return novel_caps


def get_mocap_path(opt):
    if os.path.basename(opt.scene_dir) == 'bike' and opt.motion_name == 'jumpandroll':
        return './data/SFU/0012/0012_JumpAndRoll001_poses.npz', 100, 400, 30
    else:
        raise ValueError('Define new elif branch')


def get_manual_alignment(opt):
    if os.path.basename(opt.scene_dir) == 'bike' and opt.motion_name == 'jumpandroll':
        manual_trans = np.array([0.08, 0.12, 0.4])
        manual_rot = np.array([95.8, 10.4, 1.8]) / 180 * np.pi
        manual_scale = 0.14
    else:
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([0, 0, 0]) / 180 * np.pi
        manual_scale = 1
    return manual_trans, manual_rot, manual_scale


def read_human_poses(opt, scene):
    # read mocap data(already in SMPL format)
    mocap_path, start_idx, end_idx, skip = get_mocap_path(opt)
    motions = np.load(mocap_path)
    poses = motions['poses'][start_idx:end_idx:skip]
    poses = poses[:, :72]
    poses[:, 66:] = 0
    trans = motions['trans'][start_idx:end_idx:skip]
    beta = scene.smpls[0]['betas']

    body_model = SMPL(
        './data/smplx/smpl',
        gender='neutral',
        device=torch.device('cpu')
    )

    # read manual alignment
    manual_trans, manual_rot, manual_scale = get_manual_alignment(opt)
    M_R = transformations.euler_matrix(*manual_rot)
    M_S = np.eye(4)
    M_S[:3, :3] *= manual_scale
    M_T = transformations.translation_matrix(manual_trans)
    T_mocap2scene = M_T[None] @ M_S[None] @ M_R[None]

    # 大 pose
    da_smpl = np.zeros_like(np.zeros((1, 72)))
    da_smpl = da_smpl.reshape(-1, 3)
    da_smpl[1] = np.array([0, 0, 1.0])
    da_smpl[2] = np.array([0, 0, -1.0])
    da_smpl = da_smpl.reshape(1, -1)

    raw_verts = []
    Ts = []
    for i, p in enumerate(poses):
        # transformations from T-pose to mocap pose(random scale)
        _, T_t2mocap = body_model.verts_transformations(
            return_tensor=False,
            poses=p[None],
            betas=beta[None],
            transl=trans[i][None]
        )
        # transform mocap data to scene space
        T_t2scene = T_mocap2scene @ T_t2mocap
        # T-pose to Da-pose
        _, T_t2da = body_model.verts_transformations(
            return_tensor=False,
            poses=da_smpl,
            betas=beta[None]
        )
        # Da-pose to scene space
        T_da2scene = T_t2scene @ np.linalg.inv(T_t2da)
        # Da-pose verts
        temp_static_verts, _ = body_model(
            return_tensor=False,
            return_joints=True,
            poses=da_smpl,
            betas=beta[None]
        )
        # verts in scene
        verts = np.einsum('BNi, Bi->BN', T_da2scene, ray_utils.to_homogeneous(temp_static_verts))[:, :3].astype(np.float32)
        raw_verts.append(verts)
        Ts.append(T_da2scene)
    return raw_verts, Ts


def main(opt):
    # read scene
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

    # read human poses
    raw_verts, Ts = read_human_poses(opt, scene)

    # read novel captures(cameras)
    caps = read_novel_caps(opt, len(raw_verts), scene)

    # read network
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    # render
    for i in range(len(raw_verts)):
        out = render_utils.render_hybrid_nerf(
            net,
            caps[i],
            raw_verts[i],
            scene.faces,
            Ts[i],
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            geo_threshold=opt.geo_threshold,
            return_depth=False
        )
        save_path = os.path.join('./demo', f'reposing/{os.path.basename(opt.scene_dir)}', f'out_{str(i).zfill(4)}.png')
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
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')
    parser.add_argument('--motion_name', default='speedvault', type=str, help='')

    opt = parser.parse_args()

    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    main(opt)
