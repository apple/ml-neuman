# Code based on nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py
# License from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/LICENSE


import random

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras
)
from PIL import Image

from cameras.camera_pose import CameraPose
from utils import ray_utils
from utils.constant import DEFAULT_GEO_THRESH
from geometry import transformations

trans_t = lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])

rot_phi = lambda phi: np.array([
    [1, 0,           0,            0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi),  0],
    [0, 0,           0,            1]])

rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0,          1, 0,           0],
    [np.sin(th), 0, np.cos(th),  0],
    [0,          0, 0,           1]])


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    transf = np.array([
        [1, 0,  0,  0],
        [0, -1, 0,  0],
        [0, 0,  -1, 0],
        [0, 0,  0,  1.],
    ])
    c2w = c2w @ transf
    return CameraPose.from_camera_to_world(c2w, unstable=True)


def default_360_path(center, up, dist, res=40, rad=360):
    up2 = np.array([0, 0, 1])
    axis = np.cross(up, up2)
    angle = transformations.angle_between_vectors(up, up2)
    rot = transformations.rotation_matrix(-angle, axis)
    trans = transformations.translation_matrix(center)

    poses = [pose_spherical(angle, 0, dist) for angle in np.linspace(-rad / 2, rad / 2, res + 1)[:-1]]
    poses = [CameraPose.from_camera_to_world(trans @ rot @ p.camera_to_world) for p in poses]
    return poses


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkg=True):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device = raw.device
    _raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = (z_vals[..., 1:] - z_vals[..., :-1])
    dists = torch.cat([dists, torch.Tensor([1e10], ).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=device) * raw_noise_std
    alpha = _raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkg:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_vanilla(coarse_net, cap, fine_net=None, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, near_far_source='bkg', return_depth=False, ablate_nerft=False):
    device = next(coarse_net.parameters()).device

    def build_batch(origins, dirs, near, far):
        _bs = origins.shape[0]
        ray_batch = {
            'origin':    torch.from_numpy(origins).float().to(device),
            'direction': torch.from_numpy(dirs).float().to(device),
            'near':      torch.tensor([near] * _bs, dtype=torch.float32)[..., None].to(device),
            'far':       torch.tensor([far] * _bs, dtype=torch.float32)[..., None].to(device)
        }
        return ray_batch

    with torch.set_grad_enabled(False):
        origins, dirs = ray_utils.shot_all_rays(cap)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                cap.near[near_far_source],
                cap.far[near_far_source]
            )
            if ablate_nerft:
                cur_time = cap.frame_id['frame_id'] / cap.frame_id['total_frames']
                coarse_time = torch.ones(origins[i:i + rays_per_batch].shape[0], samples_per_ray, 1, device=device) * cur_time
                fine_time = torch.ones(origins[i:i + rays_per_batch].shape[0], samples_per_ray + importance_samples_per_ray, 1, device=device) * cur_time
            else:
                coarse_time, fine_time = None, None
            _pts, _dirs, _z_vals = ray_utils.ray_to_samples(ray_batch, samples_per_ray, device=device, append_t=coarse_time)
            out = coarse_net(
                _pts,
                _dirs
            )
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)

            if fine_net is not None:
                _pts, _dirs, _z_vals = ray_utils.ray_to_importance_samples(ray_batch, _z_vals, weights, importance_samples_per_ray, device=device, append_t=fine_time)
                out = fine_net(
                    _pts,
                    _dirs
                )
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)

            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
        total_rgb_map = torch.cat(total_rgb_map).reshape(*cap.shape, -1).detach().cpu().numpy()
        total_depth_map = torch.cat(total_depth_map).reshape(*cap.shape).detach().cpu().numpy()
    if return_depth:
        return total_rgb_map, total_depth_map
    return total_rgb_map


def render_smpl_nerf(net, cap, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, white_bkg=True, render_can=False, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False, return_mask=False, interval_comp=1.0):
    device = next(net.parameters()).device

    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch

    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        origins, dirs = torch.from_numpy(origins).to(device), torch.from_numpy(dirs).to(device)
        posed_verts = torch.from_numpy(posed_verts).to(device)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        total_acc_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            rgb_map = torch.zeros_like(origins[i:i + rays_per_batch]).to(device)
            depth_map = torch.zeros_like(origins[i:i + rays_per_batch, 0]).to(device)
            acc_map = torch.zeros_like(origins[i:i + rays_per_batch, 0]).to(device)
            temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], posed_verts, geo_threshold)
            if (temp_near >= temp_far).any():
                if white_bkg:
                    rgb_map[temp_near >= temp_far] = 1.0
                else:
                    rgb_map[temp_near >= temp_far] = 0.0
                depth_map[temp_near >= temp_far] = 0.0
                acc_map[temp_near >= temp_far] = 0.0
            if (temp_near < temp_far).any():
                ray_batch = build_batch(
                    origins[i:i + rays_per_batch][temp_near < temp_far],
                    dirs[i:i + rays_per_batch][temp_near < temp_far],
                    temp_near[temp_near < temp_far],
                    temp_far[temp_near < temp_far]
                )
                _pts, _dirs, _z_vals = ray_utils.ray_to_samples(ray_batch, samples_per_ray, device=device)
                if render_can:
                    can_pts = _pts
                    can_dirs = _dirs
                else:
                    can_pts, can_dirs, _ = ray_utils.warp_samples_to_canonical(
                        _pts.cpu().numpy(),
                        posed_verts.cpu().numpy(),
                        faces,
                        Ts
                    )
                    can_pts = torch.from_numpy(can_pts)
                    can_dirs = torch.from_numpy(can_dirs) 
                can_pts = can_pts.to(device).float()
                can_dirs = can_dirs.to(device).float()
                out = net.coarse_human_net(can_pts, can_dirs)
                out[..., -1] *= interval_comp
                _rgb_map, _, _acc_map, _, _depth_map = raw2outputs(out, _z_vals, _dirs[:, 0, :], white_bkg=white_bkg)
                rgb_map[temp_near < temp_far] = _rgb_map
                depth_map[temp_near < temp_far] = _depth_map
                acc_map[temp_near < temp_far] = _acc_map
            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
            total_acc_map.append(acc_map)
        total_rgb_map = torch.cat(total_rgb_map).reshape(*cap.shape, -1).detach().cpu().numpy()
        total_depth_map = torch.cat(total_depth_map).reshape(*cap.shape).detach().cpu().numpy()
        total_acc_map = torch.cat(total_acc_map).reshape(*cap.shape).detach().cpu().numpy()
    if return_depth and return_mask:
        return total_rgb_map, total_depth_map, total_acc_map
    if return_depth:
        return total_rgb_map, total_depth_map
    if return_mask:
        return total_rgb_map, total_acc_map
    return total_rgb_map


def render_hybrid_nerf(net, cap, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False):
    device = next(net.parameters()).device

    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch

    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        total_acc_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            rgb_map = np.zeros_like(origins[i:i + rays_per_batch])
            depth_map = np.zeros_like(origins[i:i + rays_per_batch, 0])
            acc_map = np.zeros_like(origins[i:i + rays_per_batch, 0])
            bkg_ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                np.array([cap.near['bkg']] * origins[i:i + rays_per_batch].shape[0]),
                np.array([cap.far['bkg']] * origins[i:i + rays_per_batch].shape[0]),
            )
            bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_samples(bkg_ray_batch, samples_per_ray, device=device)
            bkg_out = net.coarse_bkg_net(
                bkg_pts,
                bkg_dirs
            )
            if net.fine_bkg_net is not None:
                _, _, _, bkg_weights, _ = raw2outputs(bkg_out, bkg_z_vals, bkg_dirs[:, 0, :], white_bkg=white_bkg)
                bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_importance_samples(bkg_ray_batch, bkg_z_vals, bkg_weights, importance_samples_per_ray, device=device)
                bkg_out = net.fine_bkg_net(
                    bkg_pts,
                    bkg_dirs
                )
            temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], posed_verts, geo_threshold)
            if (temp_near >= temp_far).any():
                # no fuse
                # render bkg colors
                coarse_bkg_rgb_map, _, coarse_bkg_acc_map, weights, coarse_bkg_depth_map = raw2outputs(
                    bkg_out[temp_near >= temp_far],
                    bkg_z_vals[temp_near >= temp_far],
                    bkg_dirs[temp_near >= temp_far][:, 0, :],
                    white_bkg=white_bkg
                )
                rgb_map[temp_near >= temp_far] = coarse_bkg_rgb_map.detach().cpu().numpy()
                depth_map[temp_near >= temp_far] = coarse_bkg_depth_map.detach().cpu().numpy()
                acc_map[temp_near >= temp_far] = 0

            if (temp_near < temp_far).any():
                human_ray_batch = build_batch(
                    origins[i:i + rays_per_batch][temp_near < temp_far],
                    dirs[i:i + rays_per_batch][temp_near < temp_far],
                    temp_near[temp_near < temp_far],
                    temp_far[temp_near < temp_far]
                )
                human_pts, human_dirs, human_z_vals = ray_utils.ray_to_samples(human_ray_batch, samples_per_ray, device=device)
                can_pts, can_dirs, _ = ray_utils.warp_samples_to_canonical(
                    human_pts.cpu().numpy(),
                    posed_verts,
                    faces,
                    Ts
                )
                can_pts = torch.from_numpy(can_pts).to(device).float()
                can_dirs = torch.from_numpy(can_dirs).to(device).float()
                human_out = net.coarse_human_net(can_pts, can_dirs)
                coarse_total_zvals, coarse_order = torch.sort(torch.cat([bkg_z_vals[temp_near < temp_far], human_z_vals], -1), -1)
                coarse_total_out = torch.cat([bkg_out[temp_near < temp_far], human_out], 1)
                _b, _n, _c = coarse_total_out.shape
                coarse_total_out = coarse_total_out[
                    torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                    coarse_order.view(_b, _n, 1).repeat(1, 1, _c),
                    torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
                ]
                human_rgb_map, _, _, _, human_depth_map = raw2outputs(
                    coarse_total_out,
                    coarse_total_zvals,
                    human_dirs[:, 0, :],
                    white_bkg=white_bkg,
                )

                _, _, human_acc_map, _, _ = raw2outputs(
                    human_out,
                    human_z_vals,
                    human_dirs[:, 0, :],
                    white_bkg=white_bkg,
                )
                rgb_map[temp_near < temp_far] = human_rgb_map.detach().cpu().numpy()
                depth_map[temp_near < temp_far] = human_depth_map.detach().cpu().numpy()
                acc_map[temp_near < temp_far] = human_acc_map.detach().cpu().numpy()
            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
            total_acc_map.append(acc_map)
        total_rgb_map = np.concatenate(total_rgb_map).reshape(*cap.shape, -1)
        total_depth_map = np.concatenate(total_depth_map).reshape(*cap.shape)
        total_acc_map = np.concatenate(total_acc_map).reshape(*cap.shape)
    if return_depth:
        return total_rgb_map, total_depth_map
    return total_rgb_map


def render_hybrid_nerf_multi_persons(bkg_model, cap, human_models, posed_verts, faces, Ts, rays_per_batch=32768, samples_per_ray=64, importance_samples_per_ray=128, white_bkg=True, geo_threshold=DEFAULT_GEO_THRESH, return_depth=False):
    device = next(bkg_model.parameters()).device

    def build_batch(origins, dirs, near, far):
        if isinstance(origins, torch.Tensor):
            ray_batch = {
                'origin':    (origins).float().to(device),
                'direction': (dirs).float().to(device),
                'near':      (near[..., None]).float().to(device),
                'far':       (far[..., None]).float().to(device)
            }
        else:
            ray_batch = {
                'origin':    torch.from_numpy(origins).float().to(device),
                'direction': torch.from_numpy(dirs).float().to(device),
                'near':      torch.from_numpy(near[..., None]).float().to(device),
                'far':       torch.from_numpy(far[..., None]).float().to(device)
            }
        return ray_batch
    with torch.set_grad_enabled(False):
        coords = np.argwhere(np.ones(cap.shape))[:, ::-1]
        origins, dirs = ray_utils.shot_rays(cap, coords)
        total_rays = origins.shape[0]
        total_rgb_map = []
        total_depth_map = []
        for i in range(0, total_rays, rays_per_batch):
            print(f'{i} / {total_rays}')
            bkg_ray_batch = build_batch(
                origins[i:i + rays_per_batch],
                dirs[i:i + rays_per_batch],
                np.array([cap.near['bkg']] * origins[i:i + rays_per_batch].shape[0]),
                np.array([cap.far['bkg']] * origins[i:i + rays_per_batch].shape[0]),
            )
            bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_samples(bkg_ray_batch, samples_per_ray, device=device)
            bkg_out = bkg_model.coarse_bkg_net(
                bkg_pts,
                bkg_dirs
            )
            if bkg_model.fine_bkg_net is not None:
                _, _, _, bkg_weights, _ = raw2outputs(bkg_out, bkg_z_vals, bkg_dirs[:, 0, :], white_bkg=white_bkg)
                bkg_pts, bkg_dirs, bkg_z_vals = ray_utils.ray_to_importance_samples(bkg_ray_batch, bkg_z_vals, bkg_weights, importance_samples_per_ray, device=device)
                bkg_out = bkg_model.fine_bkg_net(
                    bkg_pts,
                    bkg_dirs
                )
            human_out_dict = {
                'out': [],
                'z_val': [],
            }
            for _net, _posed_verts, _faces, _Ts in zip(human_models, posed_verts, faces, Ts):
                temp_near, temp_far = ray_utils.geometry_guided_near_far(origins[i:i + rays_per_batch], dirs[i:i + rays_per_batch], _posed_verts, geo_threshold)
                # generate ray samples
                num_empty_rays = bkg_pts.shape[0]
                empty_human_out = torch.zeros([num_empty_rays, samples_per_ray, 4], device=device)
                empty_human_z_vals = torch.stack([torch.linspace(cap.far['bkg'] * 2, cap.far['bkg'] * 3, samples_per_ray)] * num_empty_rays).to(device)
                if (temp_near < temp_far).any():
                    human_ray_batch = build_batch(
                        origins[i:i + rays_per_batch][temp_near < temp_far],
                        dirs[i:i + rays_per_batch][temp_near < temp_far],
                        temp_near[temp_near < temp_far],
                        temp_far[temp_near < temp_far]
                    )
                    human_pts, human_dirs, human_z_vals = ray_utils.ray_to_samples(human_ray_batch, samples_per_ray, device=device)
                    can_pts, can_dirs, _ = ray_utils.warp_samples_to_canonical(
                        human_pts.cpu().numpy(),
                        _posed_verts,
                        _faces,
                        _Ts
                    )
                    can_pts = torch.from_numpy(can_pts).to(device).float()
                    can_dirs = torch.from_numpy(can_dirs).to(device).float()
                    human_out = _net.coarse_human_net(can_pts, can_dirs)
                    empty_human_out[temp_near < temp_far] = human_out
                    empty_human_z_vals[temp_near < temp_far] = human_z_vals
                human_out_dict['out'].append(empty_human_out)
                human_out_dict['z_val'].append(empty_human_z_vals)
            coarse_total_zvals, coarse_order = torch.sort(torch.cat([bkg_z_vals, torch.cat(human_out_dict['z_val'], 1)], -1), -1)
            coarse_total_out = torch.cat([bkg_out, torch.cat(human_out_dict['out'], 1)], 1)
            _b, _n, _c = coarse_total_out.shape
            coarse_total_out = coarse_total_out[
                torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
                coarse_order.view(_b, _n, 1).repeat(1, 1, _c),
                torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
            ]
            rgb_map, _, _, _, depth_map = raw2outputs(
                coarse_total_out,
                coarse_total_zvals,
                bkg_dirs[:, 0, :],
                white_bkg=white_bkg,
            )
            total_rgb_map.append(rgb_map)
            total_depth_map.append(depth_map)
        total_rgb_map = torch.cat(total_rgb_map).reshape(*cap.shape, -1).detach().cpu().numpy()
        total_depth_map = torch.cat(total_depth_map).reshape(*cap.shape).detach().cpu().numpy()
        if return_depth:
            return total_rgb_map, total_depth_map
        return total_rgb_map


def phong_renderer_from_pinhole_cam(cam, device='cpu'):
    focal_length = torch.tensor([[cam.fx, cam.fy]])
    principal_point = torch.tensor([[cam.width - cam.cx, cam.height - cam.cy]])  # In PyTorch3D, we assume that +X points left, and +Y points up and +Z points out from the image plane.
    image_size = torch.tensor([[cam.height, cam.width]])
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, in_ndc=False, image_size=image_size, device=device)
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    raster_settings = RasterizationSettings(
        image_size=(cam.height, cam.width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return silhouette_renderer


def overlay_smpl(img, verts, faces, cap):
    device = verts.device
    renderer = phong_renderer_from_pinhole_cam(cap.pinhole_cam, device=device)
    R = torch.from_numpy(cap.cam_pose.rotation_matrix[:3, :3].T)[None].to(device)
    T = torch.from_numpy(cap.cam_pose.translation_vector)[None].to(device)
    mesh_col = torch.ones_like(verts)[None].to(device)
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=mesh_col)
    )
    silhouette = renderer(meshes_world=mesh, R=R, T=T)
    silhouette = torch.rot90(silhouette[0].cpu().detach(), k=2).numpy()
    silhouette = Image.fromarray(np.uint8(silhouette * 255))
    bkg = Image.fromarray(np.concatenate([img, np.ones_like(img[..., 0:1]) * 255], axis=-1))
    overlay = Image.alpha_composite(bkg, silhouette)
    return np.array(overlay)[..., :3]
