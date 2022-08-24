# Code based on nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
# License from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/LICENSE


import numpy as np
import torch
import igl

from geometry.pcd_projector import PointCloudProjectorNp
from utils.constant import DEFAULT_GEO_THRESH, PERTURB_EPSILON


def shot_ray(cap, x, y):
    z = np.array([[1]])
    xy = np.array([[x, y]])
    pcd_3d = PointCloudProjectorNp.pcd_2d_to_pcd_3d(xy, z, cap.intrinsic_matrix, cam2world=cap.cam_pose.camera_to_world)[0].astype(np.float32)
    orig = cap.cam_pose.camera_center_in_world
    dir = pcd_3d - orig
    dir = dir / np.linalg.norm(dir)
    return orig, dir


def shot_rays(cap, xys):
    z = np.ones((xys.shape[0], 1))
    pcd_3d = PointCloudProjectorNp.pcd_2d_to_pcd_3d(xys, z, cap.intrinsic_matrix, cam2world=cap.cam_pose.camera_to_world).astype(np.float32)
    orig = np.stack([cap.cam_pose.camera_center_in_world] * xys.shape[0])
    dir = pcd_3d - orig
    dir = dir / np.linalg.norm(dir, axis=1, keepdims=True)
    return orig, dir


def shot_all_rays(cap):
    '''
    set flip to True when use pretrained weights
    '''
    c2w = cap.cam_pose.camera_to_world
    temp_pcd = PointCloudProjectorNp.img_to_pcd_3d(np.ones(cap.size), cap.intrinsic_matrix, img=None, cam2world=c2w)
    dirs = temp_pcd - cap.cam_pose.camera_center_in_world
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origs = np.stack([cap.cam_pose.camera_center_in_world] * dirs.shape[0], axis=0)
    return origs, dirs


def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


def warp_samples_to_canonical(pts, verts, faces, T, return_T=False, dist2=None, f_id=None, closest=None, out_cuda=False):
    if dist2 is None:
        dist2, f_id, closest = igl.point_mesh_squared_distance(pts, verts, faces[:, :3])
    closest_tri = verts[faces[:, :3][f_id]]
    barycentric = igl.barycentric_coordinates_tri(closest, closest_tri[:, 0, :].copy(), closest_tri[:, 1, :].copy(), closest_tri[:, 2, :].copy())
    T_interp = (T[faces[:, :3][f_id]] * barycentric[..., None, None]).sum(axis=1)
    if out_cuda:
        T_interp_inv = torch.inverse(torch.from_numpy(T_interp).float().to('cuda'))
    else:
        T_interp_inv = np.linalg.inv(T_interp)
    if return_T:
        return T_interp_inv, f_id, dist2
    new_pts = (T_interp_inv @ to_homogeneous(pts)[..., None])[:, :3, 0]
    new_dirs = new_pts[1:] - new_pts[:-1]
    new_dirs = np.concatenate([new_dirs, new_dirs[-1:]])
    new_dirs = new_dirs / np.linalg.norm(new_dirs, axis=1, keepdims=True)

    return new_pts, new_dirs, closest


def warp_samples_to_canonical_diff(pts, verts, faces, T):
    signed_dist, f_id, closest = igl.signed_distance(pts, verts.detach().cpu().numpy(), faces[:, :3])

    # differentiable barycentric interpolation
    closest_tri = verts[faces[:, :3][f_id]]
    closest = torch.from_numpy(closest).float().to(verts.device)
    v0v1 = closest_tri[:, 1] - closest_tri[:, 0]
    v0v2 = closest_tri[:, 2] - closest_tri[:, 0]
    v1v2 = closest_tri[:, 2] - closest_tri[:, 1]
    v2v0 = closest_tri[:, 0] - closest_tri[:, 2]
    v1p = closest - closest_tri[:, 1]
    v2p = closest - closest_tri[:, 2]
    N = torch.cross(v0v1, v0v2)
    denom = torch.bmm(N.unsqueeze(dim=1), N.unsqueeze(dim=2)).squeeze()
    C1 = torch.cross(v1v2, v1p)
    u = torch.bmm(N.unsqueeze(dim=1), C1.unsqueeze(dim=2)).squeeze() / denom
    C2 = torch.cross(v2v0, v2p)
    v = torch.bmm(N.unsqueeze(dim=1), C2.unsqueeze(dim=2)).squeeze() / denom
    w = 1 - u - v
    barycentric = torch.stack([u, v, w], dim=1)

    T_interp = (T[faces[:, :3][f_id]] * barycentric[..., None, None]).sum(axis=1)
    T_interp_inv = torch.inverse(T_interp)

    return T_interp_inv, f_id, signed_dist


def ray_to_samples(ray_batch,
                   samples_per_ray,
                   lindisp=False,
                   perturb=0.,
                   device='cpu',
                   append_t=None
                   ):
    '''
    reference: https://github.com/yenchenlin/nerf-pytorch
    '''
    rays_per_batch = ray_batch['origin'].shape[0]
    rays_o, rays_d = ray_batch['origin'], ray_batch['direction']  # [rays_per_batch, 3] each
    near, far = ray_batch['near'], ray_batch['far']  # [-1,1]
    assert near.shape[0] == far.shape[0] == rays_per_batch

    t_vals = torch.linspace(0., 1., steps=samples_per_ray, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.clip(
            torch.rand(z_vals.shape, device=device),
            min=PERTURB_EPSILON,
            max=1-PERTURB_EPSILON
        )

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [rays_per_batch, samples_per_ray, 3]
    dirs = torch.stack([rays_d] * samples_per_ray, axis=1)
    if append_t is not None:
        pts = torch.cat([pts, append_t.to(device)], dim=-1)
    return pts, dirs, z_vals


def ray_to_importance_samples(ray_batch,
                              z_vals,
                              weights,
                              importance_samples_per_ray,
                              device='cpu',
                              including_old=True,
                              append_t=None
                              ):
    rays_o, rays_d = ray_batch['origin'].to(device), ray_batch['direction'].to(device)

    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], importance_samples_per_ray, det=True, device=device)
    z_samples = z_samples.detach()
    if including_old:
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    else:
        z_vals = z_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    total_samples = pts.shape[1]
    dirs = torch.stack([rays_d] * total_samples, axis=1)
    if append_t is not None:
        pts = torch.cat([pts, append_t.to(device)], dim=-1)
    return pts, dirs, z_vals


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, device='cpu'):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def geometry_guided_near_far(orig, dir, vert, geo_threshold):
    if isinstance(orig, torch.Tensor):
        return geometry_guided_near_far_torch(orig, dir, vert, geo_threshold)
    elif isinstance(orig, np.ndarray):
        return geometry_guided_near_far_np(orig, dir, vert, geo_threshold)


def geometry_guided_near_far_torch(orig, dir, vert, geo_threshold=DEFAULT_GEO_THRESH):
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    orig_ = torch.repeat_interleave(orig[:, None, :], num_vert, 1)
    dir_ = torch.repeat_interleave(dir[:, None, :], num_vert, 1)
    vert_ = torch.repeat_interleave(vert[None, ...], num_rays, 0)
    orig_v = vert_ - orig_
    z0 = torch.einsum('ij,ij->i', orig_v.reshape(-1, 3), dir_.reshape(-1, 3)).reshape(num_rays, num_vert)
    dz = torch.sqrt(geo_threshold**2 - (torch.norm(orig_v, dim=2)**2 - z0**2))
    near = z0 - dz
    near[near != near] = float('inf')
    near = near.min(dim=1)[0]
    far = z0 + dz
    far[far != far] = float('-inf')
    far = far.max(dim=1)[0]
    return near, far


def geometry_guided_near_far_np(orig, dir, vert, geo_threshold=DEFAULT_GEO_THRESH):
    num_vert = vert.shape[0]
    num_rays = orig.shape[0]
    orig_ = np.repeat(orig[:, None, :], num_vert, 1)
    dir_ = np.repeat(dir[:, None, :], num_vert, 1)
    vert_ = np.repeat(vert[None, ...], num_rays, 0)
    orig_v = vert_ - orig_
    z0 = np.einsum('ij,ij->i', orig_v.reshape(-1, 3), dir_.reshape(-1, 3)).reshape(num_rays, num_vert)
    dz = np.sqrt(geo_threshold**2 - (np.linalg.norm(orig_v, axis=2)**2 - z0**2))
    near = np.nan_to_num(z0-dz, nan=np.inf).min(axis=1)
    far = np.nan_to_num(z0+dz, nan=-np.inf).max(axis=1)
    return near, far
