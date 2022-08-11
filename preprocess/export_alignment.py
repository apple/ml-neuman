#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Alignment the estimated SMPL mesh(ROMP) to the reconstructed sparse scene(COLMAP).
'''
import sys
sys.path.append('..')

import argparse
import os
import copy

import torch
import joblib
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

from data_io import colmap_helper
from utils import debug_utils, ray_utils
from cameras import camera_pose
from geometry.basics import Translation, Rotation
from geometry import transformations


def read_vibe_estimates(vibe_output_path):
    vibe_estimates = joblib.load(vibe_output_path)[1]
    return vibe_estimates


def dump_romp_estimates(romp_output_dir, dump_path, scene=None):
    if os.path.isfile(dump_path):
        return
    vibe_estimates = {
        'verts': [],
        'joints3d': [],
        'joints2d_img_coord': [],
        'pose': [],
        'betas': [],
    }

    for cur, dirs, files in os.walk(romp_output_dir):
        for file in sorted(files):
            if file.endswith('npz'):
                cur_res = np.load(os.path.join(cur, file), allow_pickle=True)['results']
                assert len(cur_res) == 1
                cur_res = cur_res[0]
                vibe_estimates['verts'].append(cur_res['verts'])
                vibe_estimates['joints3d'].append(cur_res['j3d_all54'])
                vibe_estimates['joints2d_img_coord'].append(cur_res['pj2d_org'])
                vibe_estimates['pose'].append(cur_res['poses'])
                vibe_estimates['betas'].append(cur_res['betas'])
        break

    for k, v in vibe_estimates.items():
        vibe_estimates[k] = np.array(v).astype(np.float32)

    vibe_results = {}
    vibe_results[1] = vibe_estimates

    joblib.dump(vibe_results, dump_path)
    print(f'dumped ROMP results to pkl at {dump_path}')


def read_smpl(opt, scene=None):
    if opt.smpl_estimator == 'vibe':
        return read_vibe_estimates(opt.raw_smpl)
    elif opt.smpl_estimator == 'romp':
        assert os.path.isdir(opt.raw_smpl)
        dump_path = os.path.abspath(os.path.join(opt.raw_smpl, '../smpl_output_romp.pkl'))
        dump_romp_estimates(opt.raw_smpl, dump_path, scene)
        return read_vibe_estimates(dump_path)


def solve_translation(p3d, p2d, mvp):
    p3d = torch.from_numpy(p3d.copy()).float()
    p2d = torch.from_numpy(p2d.copy()).float()
    mvp = torch.from_numpy(mvp.copy()).float()
    translation = torch.zeros_like(p3d[0:1, 0:3], requires_grad=True)
    optim_list = [
        {"params": translation, "lr": 1e-3},
    ]
    optim = torch.optim.Adam(optim_list)

    total_iters = 1000
    for i in tqdm(range(total_iters), total=total_iters):
        xyzw = torch.cat([p3d[:, 0:3] + translation, torch.ones_like(p3d[:, 0:1])], axis=1)
        camera_points = torch.matmul(mvp, xyzw.T).T
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(image_points, p2d)
        loss.backward()
        optim.step()
    print('loss', loss, 'translation', translation)
    return translation.clone().detach().cpu().numpy()


def solve_scale(joints_world, cap, plane_model):
    cam_center = cap.cam_pose.camera_center_in_world
    a, b, c, d = plane_model
    scales = []
    for j in joints_world:
        jx, jy, jz = j
        # from open3d plane model is: a*x + b*y + c*z + d = 0
        # and a^2 + b^2 + c^2 = 1
        # We can convert the scale problem into a ray-plane intersection problem:
        # reference: https://education.siggraph.org/static/HyperGraph/raytrace/rayplane_intersection.htm
        # Shoting a ray from camera center, (c_x, c_y, c_z), passing through joint, (j_x, j_y, j_z), and
        # Intersecting the plane, a*x + b*y + c*z + d = 0, at some point (x, y, z)
        # Let R0 = (c_x, c_y, c_z)
        #     Rd = (j_x-c_x, j_y-c_y, j_z-c_z)
        # The ray can be written as: R(s) = R0 + s * Rd
        # with the plane equation:
        # a*(c_x + s*(j_x-c_x)) + b*(c_y + s*(j_y-c_y)) + c*(c_z + s*(j_z-c_z)) + d = 0
        # s = -(a*c_x + b*c_y + c*c_z + d) / (a*(j_x-c_x) + b*(j_y-c_y) + c*(j_z-c_z))
        # let right = -(a*c_x + b*c_y + c*c_z + d)
        #       coe = a*(j_x-c_x) + b*(j_y-c_y) + c*(j_z-c_z)
        right = -(a*cam_center[0] + b*cam_center[1] + c*cam_center[2] + d)
        coe = a*(jx-cam_center[0]) + b*(jy-cam_center[1]) + c*(jz-cam_center[2])
        s = right / coe
        if s > 0:
            scales.append(s)
    return min(scales)


def solve_transformation(verts, j3d, j2d, plane_model, colmap_cap, smpl_cap):
    mvp = np.matmul(smpl_cap.intrinsic_matrix, smpl_cap.extrinsic_matrix)
    trans = solve_translation(j3d, j2d, mvp)
    smpl_cap.cam_pose.camera_center_in_world -= trans[0]
    joints_world = (ray_utils.to_homogeneous(j3d) @ smpl_cap.cam_pose.world_to_camera.T @ colmap_cap.cam_pose.camera_to_world.T)[:, :3]
    scale = solve_scale(joints_world, colmap_cap, plane_model)
    print('scale', scale)
    transf = smpl_cap.cam_pose.world_to_camera.T * scale
    transf[3, 3] = 1
    transf = transf @ colmap_cap.cam_pose.camera_to_world_3x4.T
    verts_world = ray_utils.to_homogeneous(verts) @ transf
    return transf, verts_world


def main(opt):
    scene = colmap_helper.ColmapAsciiReader.read_scene(
        opt.scene_dir,
        opt.images_dir,
        order='video'
    )
    raw_smpl = read_smpl(opt, scene)

    assert len(raw_smpl['pose']) == len(scene.captures)

    # estimate the ground
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)

    plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)
    pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)
    inliers = np.abs(np.sum(np.multiply(scene.point_cloud[:, :3], plane_model[:3]), axis=1) + plane_model[3]) < 0.02
    inliers = list(np.where(inliers)[0])
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    # solve the alignment
    alignments = {}
    for i, cap in tqdm(enumerate(scene.captures), total=len(scene.captures)):
        pts_3d = raw_smpl['joints3d'][i]
        pts_2d = raw_smpl['joints2d_img_coord'][i]
        _, R_rod, t, inl = cv2.solvePnPRansac(pts_3d, pts_2d, cap.pinhole_cam.intrinsic_matrix, np.zeros(4), flags=cv2.SOLVEPNP_EPNP)
        t = t.astype(np.float32)[:, 0]
        R, _ = cv2.Rodrigues(R_rod)
        quat = transformations.quaternion_from_matrix(R).astype(np.float32)

        smpl_cap = copy.deepcopy(cap)
        smpl_cam_pose = camera_pose.CameraPose(Translation(t), Rotation(quat))
        smpl_cap.cam_pose = smpl_cam_pose

        # refine the translation and solve the scale
        transf, _ = solve_transformation(
            raw_smpl['verts'][i],
            raw_smpl['joints3d'][i],
            raw_smpl['joints2d_img_coord'][i],
            plane_model,
            cap,
            smpl_cap
        )
        alignments[os.path.basename(cap.image_path)] = transf
    save_path = os.path.abspath(os.path.join(opt.scene_dir, '../alignments.npy'))
    np.save(save_path, alignments)
    print(f'alignment matrix saved at: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, default=None, required=True)
    parser.add_argument('--images_dir', type=str, default=None, required=True)
    parser.add_argument('--raw_smpl', type=str, default=None, required=True)
    parser.add_argument('--smpl_estimator', type=str, default=None, required=True)
    opt = parser.parse_args()
    main(opt)
