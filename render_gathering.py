#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render multiple people in a scene.
We manually align the human to the scene using Blender, and hard code the alignments.
'''


import os
import argparse
import copy
import joblib

import imageio
import torch
import numpy as np

from geometry import transformations
from models import human_nerf
from utils import render_utils, utils, ray_utils
from data_io import neuman_helper
from options import options
from models.smpl import SMPL


ACTOR_WEIGHTS_DICT = {
    'seattle': 'seattle_human',
    'citron': 'citron_human',
    'parkinglot': 'parkinglot_human',
}


def read_novel_caps(opt, num_caps, scene):
    novel_caps = []
    if os.path.basename(opt.scene_dir) == 'seattle' and opt.motion_name == 'dance_together':
        for i in range(num_caps):
            cap = copy.deepcopy(scene.captures[20])
            ellipse_a = 0.15
            ellipse_b = 0.05
            x_offset = cap.cam_pose.right * ellipse_a * np.cos((i/num_caps) * (4*np.pi))
            y_offset = cap.cam_pose.up * ellipse_b * np.sin((i/num_caps) * (4*np.pi))
            cap.cam_pose.camera_center_in_world = cap.cam_pose.camera_center_in_world + x_offset + y_offset
            novel_caps.append(cap)
    return novel_caps


def get_mocap_path(motion_name, actor_name=None):
    if motion_name == 'dance_together':
        return './data/SFU/0018/0018_XinJiang002_poses.npz', 0, 800, 4
    else:
        raise ValueError('Define new elif branch')


def get_manual_alignment(motion_name, actor_name):
    if motion_name == 'dance_together':
        if actor_name == 'seattle':
            manual_trans = np.array([0, 0.15, 0.77])
            manual_rot = np.array([90.4, -10.9, 4]) / 180 * np.pi
            manual_scale = 0.2
        if actor_name == 'citron':
            manual_trans = np.array([-0.36, 0.13, 0.92])
            manual_rot = np.array([90, -9.4, 4]) / 180 * np.pi
            manual_scale = 0.2
        if actor_name == 'parkinglot':
            manual_trans = np.array([0.32, 0.12, 0.96])
            manual_rot = np.array([90, -11.6, 4]) / 180 * np.pi
            manual_scale = 0.2
    else:
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([0, 0, 0]) / 180 * np.pi
        manual_scale = 1
    return manual_trans, manual_rot, manual_scale


def read_actor(opt, actor_name):
    # read network
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(f'./out/{ACTOR_WEIGHTS_DICT[actor_name]}/checkpoint.pth.tar', map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    # read mocap data(already in SMPL format)
    mocap_path, start_idx, end_idx, skip = get_mocap_path(opt.motion_name)
    motions = np.load(mocap_path)
    poses = motions['poses'][start_idx:end_idx:skip]
    poses = poses[:, :72]
    poses[:, 66:] = 0
    trans = motions['trans'][start_idx:end_idx:skip]
    # read smpl betas from original source
    smpl_path = os.path.join(os.path.join(os.path.dirname(opt.scene_dir), actor_name), f'smpl_output_optimized.pkl')
    raw_smpl = joblib.load(smpl_path)
    raw_smpl = raw_smpl[list(raw_smpl.keys())[0]]
    beta = np.array(raw_smpl['betas']).mean(0)

    body_model = SMPL(
        './data/smplx/smpl',
        gender='neutral',
        device=torch.device('cpu')
    )

    # read manual alignment
    manual_trans, manual_rot, manual_scale = get_manual_alignment(opt.motion_name, actor_name)
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
    return net, raw_verts, Ts


def read_actors(opt):
    nets_list, verts_list, Ts_list = [], [], []
    for actor in opt.actors:
        net, raw_verts, Ts = read_actor(opt, actor)
        nets_list.append(net)
        verts_list.append(raw_verts)
        Ts_list.append(Ts)
    verts_list = np.array(verts_list)
    Ts_list = np.array(Ts_list)
    return nets_list, verts_list, Ts_list


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

    # read all_actors
    nets_list, verts_list, Ts_list = read_actors(opt)
    # read novel captures(cameras)
    caps = read_novel_caps(opt, len(verts_list[0]), scene)

    # read background network
    bkg_net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(bkg_net, weights['hybrid_model_state_dict'])

    # render
    for i in range(len(caps)):
        out = render_utils.render_hybrid_nerf_multi_persons(
            bkg_net,
            caps[i],
            nets_list,
            verts_list[:, i, ...],
            [scene.faces] * len(nets_list),
            Ts_list[:, i, ...],
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            geo_threshold=opt.geo_threshold,
            return_depth=False
        )
        save_path = os.path.join('./demo', f'gathering/{opt.motion_name}', f'out_{str(i).zfill(4)}.png')
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
    parser.add_argument('--motion_name', default='dance_together', type=str, help='')
    parser.add_argument('--actors', nargs="*", type=str, default=['seattle', 'citron', 'parkinglot'])

    opt = parser.parse_args()

    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)

    main(opt)
