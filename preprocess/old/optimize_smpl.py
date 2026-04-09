#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Optimize SMPL parameters.
Optimization objective: mse(rendered mask, detected mask(DensePose)).
'''

import sys
sys.path.append('..')

import os
import argparse

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, PerspectiveCameras
)
from tqdm import tqdm
import joblib
import imageio

from data_io import neuman_helper
from utils import debug_utils, utils, ray_utils, render_utils
from models.smpl import SMPL
from geometry import pcd_projector


def coco_to_smpl(coco2d):
    '''
    input 2d joints in coco dataset format,
    and out 2d joints in SMPL format.
    Non-overlapping joints are set to 0s. 
    '''
    assert coco2d.shape == (17, 2)
    smpl2d = np.zeros((24, 2))
    smpl2d[1]  = coco2d[11] # leftUpLeg
    smpl2d[2]  = coco2d[12] # rightUpLeg
    smpl2d[4]  = coco2d[13] # leftLeg
    smpl2d[5]  = coco2d[14] # rightLeg
    smpl2d[7]  = coco2d[15] # leftFoot
    smpl2d[8]  = coco2d[16] # rightFoot
    smpl2d[16] = coco2d[5]  # leftArm
    smpl2d[17] = coco2d[6]  # rightArm
    smpl2d[18] = coco2d[7]  # leftForeArm
    smpl2d[19] = coco2d[8]  # rightForeArm
    smpl2d[20] = coco2d[9]  # leftHand
    smpl2d[21] = coco2d[10] # rightHand
    return smpl2d


def densepose_name_to_idx():
    return {
        'Torso': [1, 2],
        'Right Hand': [3],
        'Left Hand': [4],
        'Left Foot': [5],
        'Right Foot': [6],
        'Upper Leg Right': [7, 9],
        'Upper Leg Left': [8, 10],
        'Lower Leg Right': [11, 13],
        'Lower Leg Left': [12, 14],
        'Upper Arm Left': [15, 17],
        'Upper Arm Right': [16, 18],
        'Lower Arm Left': [19, 21],
        'Lower Arm Right': [20, 22],
        'Head': [23, 24]
    }


def densepose_idx_to_name():
    name2idx = densepose_name_to_idx()
    idx2name = {}
    for k, v in name2idx.items():
        for item in v:
            idx2name[item] = k
    return idx2name


def silhouette_renderer_from_pinhole_cam(cam, device='cpu'):
    focal_length = torch.tensor([[cam.fx, cam.fy]])
    principal_point = torch.tensor([[cam.width - cam.cx, cam.height - cam.cy]])  # In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane.
    image_size = torch.tensor([[cam.height, cam.width]])
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=image_size, device=device)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(cam.height, cam.width),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    return silhouette_renderer


def vertext_forward(pose, betas, align, body_model, scale):
    device = pose.device

    T_pose = torch.zeros_like(pose)
    T_pose = T_pose.reshape(-1, 3)

    _, mesh_transf = body_model.verts_transformations(
        return_tensor=True,
        poses=pose[None],
        betas=betas[None],
        transl=torch.zeros([1, 3]).float().to(device),
        concat_joints=True
    )
    mesh_transf = align.T @ mesh_transf
    s = torch.eye(4).to(mesh_transf.device)
    s[:3, :3] *= scale
    mesh_transf = s @ mesh_transf
    T_pose_verts, T_pose_joints = body_model(return_tensor=True,
                                               return_joints=True,
                                               poses=T_pose[None],
                                               betas=betas[None],
                                               transl=torch.zeros([1, 3]).float().to(device),
                                               )
    world_verts = torch.einsum('bni, bi->bn', mesh_transf[0], ray_utils.to_homogeneous(torch.cat([T_pose_verts, T_pose_joints], dim=0)))[:, :3]
    world_verts, world_joints = world_verts[:6890, :][None], world_verts[6890:, :][None]
    return world_verts, world_joints


def turn_smpl_gradient_on(dp_mask):
    '''
    only apply gradients on certain joints.
    In our case we only optimize the limbs.
    '''
    grad_mask = np.zeros([24, 3])
    idx2name = densepose_idx_to_name()
    visible = [idx2name[i] for i in range(1, 25) if i in np.unique(dp_mask)]
    if 'Upper Leg Left' in visible:
        grad_mask[1, 0] = 1
        grad_mask[1, 2] = 1
    if 'Upper Leg Right' in visible:
        grad_mask[2, 0] = 1
        grad_mask[2, 2] = 1
    if 'Lower Leg Left' in visible:
        grad_mask[4, 0] = 1
    if 'Lower Leg Right' in visible:
        grad_mask[5, 0] = 1
    if 'Left Foot' in visible:
        grad_mask[7] = 1
    if 'Right Foot' in visible:
        grad_mask[8] = 1
    if 'Upper Arm Left' in visible:
        grad_mask[16, 1] = 1
        grad_mask[16, 2] = 1
    if 'Upper Arm Right' in visible:
        grad_mask[17, 1] = 1
        grad_mask[17, 2] = 1
    if 'Lower Arm Left' in visible:
        grad_mask[18, 1] = 1
    if 'Lower Arm Right' in visible:
        grad_mask[19, 1] = 1
    return grad_mask.reshape(-1)


def clip_smpl_vals():
    '''
    limit the pose(joint angle) changes for certain joints.
    for example, knee only has 1 degree of freedom.(in skeleton model)
    '''
    limits = np.ones([24, 3, 2])
    limits[..., 0] *= -360
    limits[..., 1] *= 360
    # knees
    limits[4, 0] = [0, 160]
    limits[4, 1] = [0, 0]
    limits[4, 2] = [0, 0]
    limits[5, 0] = [0, 160]
    limits[5, 1] = [0, 0]
    limits[5, 2] = [0, 0]
    # feet
    limits[7, 0] = [-45, 90]
    limits[7, 1] = [-60, 60]
    limits[7, 2] = [-10, 10]
    limits[8, 0] = [-45, 90]
    limits[8, 1] = [-60, 60]
    limits[8, 2] = [-10, 10]
    # elbow
    limits[18, 1] = [-160, 0]
    limits[19, 2] = [0, 160]
    return limits.reshape(-1, 2) / 180 * np.pi


def optimize_smpl(cap, smpl, faces, body_model, align, scale, num_iters=100):
    device = body_model.device
    # create renderer
    renderer = silhouette_renderer_from_pinhole_cam(cap.pinhole_cam, device=device)
    R = torch.from_numpy(cap.cam_pose.rotation_matrix[:3, :3].T)[None].to(device)
    T = torch.from_numpy(cap.cam_pose.translation_vector)[None].to(device)

    # create tensors
    pose = torch.nn.Parameter(torch.tensor(smpl['pose'], device=device).float(), requires_grad=True)
    betas = torch.nn.Parameter(torch.tensor(smpl['betas'], device=device).float(), requires_grad=False)
    temp = np.eye(4)
    temp[..., :3] = align
    align = torch.nn.Parameter(torch.tensor(temp, device=device).float(), requires_grad=False)
    mask_target = torch.from_numpy(cap.binary_mask).float().to(device)
    joints_target = cap.keypoints[:, :2]
    joints_target[cap.keypoints[:, 2] < 0.3] = 0 # discard low confidence detections
    joints_target = torch.from_numpy(
        coco_to_smpl(joints_target[:, :2])
    ).float().to(device)
    joints_mask = (joints_target.sum(dim=1) != 0)

    # create optimizer
    optim_list = [{"params": pose, "lr": 5e-3}]
    optim = torch.optim.Adam(optim_list)

    # only allow gradient w.r.t certain joints
    grad_mask = turn_smpl_gradient_on(cap.densepose)
    grad_mask = torch.from_numpy(grad_mask).float().to(device)

    limits = clip_smpl_vals()
    limits = torch.from_numpy(limits).float().to(device)

    for i in tqdm(range(num_iters), total=num_iters):
        optim.zero_grad()
        world_verts, world_joints = vertext_forward(pose, betas, align, body_model, scale)
        world_verts, world_joints = world_verts[0], world_joints[0]
        mesh = Meshes(
            verts=[world_verts],
            faces=[faces]
        )
        proj_joints = pcd_projector.pcd_3d_to_pcd_2d_torch(
            (world_joints.T)[None],
            torch.from_numpy(cap.intrinsic_matrix).float().to(device)[None],
            torch.from_numpy(cap.extrinsic_matrix).float().to(device)[None],
            torch.from_numpy(np.array(cap.shape)).float().to(device)[None],
            keep_z=False,
            norm_coord=False,
        )[0].T

        loss = torch.nn.functional.mse_loss(
            proj_joints[joints_mask], joints_target[joints_mask]
        )

        silhouette = renderer(meshes_world=mesh, R=R, T=T)
        silhouette = torch.rot90(silhouette[0, ..., 3], k=2)
        loss += torch.nn.functional.mse_loss(silhouette, mask_target)
        loss.backward()
        valid_mask = ((pose < limits[..., 1]) * (pose > limits[..., 0])).float()
        pose.grad = pose.grad * grad_mask * valid_mask
        optim.step()
    return pose.detach().cpu().numpy()


def main(opt):
    device = torch.device('cuda')
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=None,
        normalize=False,
        densepose_dir='densepose',
        smpl_type='romp'
    )
    utils.move_smpls_to_torch(scene, device)
    raw_alignments = np.load(os.path.join(opt.scene_dir, 'alignments.npy'), allow_pickle=True).item()
    faces = torch.from_numpy(scene.faces[:, :3]).to(device)

    body_model = SMPL(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl'),
        gender='neutral',
        device=device
    )
    results = {
        1: {
            'pose': [],
            'betas': [],
        }
    }
    for i in range(len(scene.captures)):
        new_smpl = optimize_smpl(
            scene.captures[i],
            scene.smpls[i],
            faces,
            body_model,
            raw_alignments[os.path.basename(scene.captures[i].image_path)],
            scene.scale
        )
        results[1]['pose'].append(new_smpl)
        results[1]['betas'].append(scene.smpls[i]['betas'])

    joblib.dump(results, os.path.abspath(os.path.join(opt.scene_dir, 'smpl_output_optimized.pkl')))


def dump_visualizations(opt):
    device = torch.device('cuda')
    for smpl_type in ['romp', 'optimized']:
        scene = neuman_helper.NeuManReader.read_scene(
            opt.scene_dir,
            tgt_size=None,
            normalize=False,
            densepose_dir='densepose',
            smpl_type=smpl_type
        )
        if not os.path.isdir(os.path.join(opt.scene_dir, f'overlays/{smpl_type}')):
            os.makedirs(os.path.join(opt.scene_dir, f'overlays/{smpl_type}'))
        for i, cap in enumerate(scene.captures):
            img = render_utils.overlay_smpl(
                cap.image,
                torch.from_numpy(scene.verts[i]).to(device),
                torch.from_numpy(scene.faces[..., :3]).to(device),
                cap
            )
            imageio.imsave(
                os.path.abspath(os.path.join(opt.scene_dir, f'overlays/{smpl_type}/{os.path.basename(cap.image_path)}')),
                img
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, default=None, required=True)
    opt = parser.parse_args()
    main(opt)
    with torch.no_grad():
        dump_visualizations(opt)
