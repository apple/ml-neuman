#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Canonical human NeRF trainer
'''

import os
import math
import time
import random

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import igl
import lpips

from utils import utils, render_utils, ray_utils
from trainers import tensorboard_helper
from options import options
from cameras.captures import ResizedPinholeCapture
from cameras.pinhole_camera import PinholeCamera
from models.vanilla import weight_reset
from utils.constant import HARD_SURFACE_OFFSET, PATCH_SIZE, PATCH_SIZE_SQUARED, CANONICAL_ZOOM_FACTOR, CANONICAL_CAMERA_DIST


LOSS_NAMES = [
    'fine_rgb_loss',
    'lpips_loss',
    'color_range_reg',
    'smpl_sym_reg',
    'smpl_shape_reg',
    'mask_loss',
    'sparsity_reg'
]


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


def turn_smpl_gradient_off(dp_mask):
    assert dp_mask is not None
    grad_mask = np.ones([24, 3])
    idx2name = densepose_idx_to_name()
    visible = [idx2name[i] for i in range(1, 25) if i in np.unique(dp_mask)]
    if 'Upper Leg Left' not in visible:
        grad_mask[1] = 0
    if 'Upper Leg Right' not in visible:
        grad_mask[2] = 0
    if 'Lower Leg Left' not in visible:
        grad_mask[4] = 0
    if 'Lower Leg Right' not in visible:
        grad_mask[5] = 0
    if 'Left Foot' not in visible:
        grad_mask[7] = 0
        grad_mask[10] = 0
    if 'Right Foot' not in visible:
        grad_mask[8] = 0
        grad_mask[11] = 0
    if 'Upper Arm Left' not in visible:
        grad_mask[16] = 0
    if 'Upper Arm Right' not in visible:
        grad_mask[17] = 0
    if 'Lower Arm Left' not in visible:
        grad_mask[18] = 0
    if 'Lower Arm Right' not in visible:
        grad_mask[19] = 0
    if 'Left Hand' not in visible:
        grad_mask[20] = 0
        grad_mask[22] = 0
    if 'Right Hand' not in visible:
        grad_mask[21] = 0
        grad_mask[23] = 0
    if 'Head' not in visible:
        grad_mask[12] = 0
        grad_mask[15] = 0
    return grad_mask.reshape(-1)


class HumanNeRFTrainer():
    def __init__(
        self,
        opt,
        net,
        optimizer,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        interval_comp=1.0,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.opt = opt
        self.net = net
        self.use_cuda = opt.use_cuda
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = opt.out
        if not os.path.exists(opt.out):
            os.makedirs(opt.out)
        self.epoch = 0
        self.iteration = 0
        self.max_iter = opt.max_iter
        self.valid_iter = opt.valid_iter
        self.tb_pusher = tensorboard_helper.TensorboardPusher(opt)
        self.push_opt_to_tb()
        self.need_resume = opt.resume
        if self.need_resume:
            self.resume()
        if self.opt.load_weights:
            self.load_pretrained_weights()
        self.penalize_smpl_alpha = opt.penalize_smpl_alpha
        self.penalize_symmetric_alpha = opt.penalize_symmetric_alpha
        self.penalize_dummy = opt.penalize_dummy
        self.penalize_hard_surface = opt.penalize_hard_surface
        self.penalize_color_range = opt.penalize_color_range
        self.penalize_outside = opt.penalize_outside
        self.penalize_mask = opt.penalize_mask
        self.penalize_lpips = opt.penalize_lpips
        self.penalize_sharp_edge = opt.penalize_sharp_edge
        if self.penalize_lpips > 0:
            self.lpips_loss_fn = lpips.LPIPS(net='alex').to(next(self.net.parameters()).device)
        self.interval_comp = interval_comp

        center, up = utils.smpl_verts_to_center_and_up(self.val_dataset.scene.static_vert[0])
        render_poses = render_utils.default_360_path(center, up, CANONICAL_CAMERA_DIST, 100)
        if opt.tgt_size is not None:
            render_size = opt.tgt_size
        else:
            render_size = self.val_dataset.scene.captures[0].pinhole_cam.shape
        self.can_caps = [ResizedPinholeCapture(
            PinholeCamera(
                self.val_dataset.scene.captures[0].pinhole_cam.width,
                self.val_dataset.scene.captures[0].pinhole_cam.height,
                CANONICAL_ZOOM_FACTOR * self.val_dataset.scene.captures[0].pinhole_cam.width,
                CANONICAL_ZOOM_FACTOR * self.val_dataset.scene.captures[0].pinhole_cam.width,
                self.val_dataset.scene.captures[0].pinhole_cam.width / 2.0,
                self.val_dataset.scene.captures[0].pinhole_cam.height / 2.0,
            ),
            rp,
            tgt_size=render_size
        ) for rp in render_poses]

    def push_opt_to_tb(self):
        opt_str = options.opt_to_string(self.opt)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_text({'options': opt_str})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def train(self):
        '''entrance of the whole training process
        '''
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch,
                                 max_epoch,
                                 desc='Train',
                                 ncols=80):
            self.epoch = epoch
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

    def _eval_bkg_samples(self, batch, device):
        bkg_batch = {
            'origin':    batch['origin'].clone(),
            'direction': batch['direction'].clone(),
            'near':      batch['bkg_near'].clone(),
            'far':       batch['bkg_far'].clone(),
        }
        coarse_bkg_samples = ray_utils.ray_to_samples(
            bkg_batch,
            self.opt.samples_per_ray,
            perturb=0
        )
        coarse_bkg_pts = coarse_bkg_samples[0].to(device)
        coarse_bkg_dirs = coarse_bkg_samples[1].to(device)
        coarse_bkg_z_vals = coarse_bkg_samples[2].to(device)
        coarse_bkg_out = self.net.coarse_bkg_net(
            coarse_bkg_pts,
            coarse_bkg_dirs
        )
        coarse_bkg_out = coarse_bkg_out.detach()

        _, _, _, coarse_bkg_weights, _ = render_utils.raw2outputs(
            coarse_bkg_out,
            coarse_bkg_z_vals,
            coarse_bkg_dirs[:, 0, :],
            white_bkg=self.opt.white_bkg
        )
        coarse_bkg_weights = coarse_bkg_weights.detach()

        fine_bkg_pts, fine_bkg_dirs, fine_bkg_z_vals = ray_utils.ray_to_importance_samples(
            batch,
            coarse_bkg_z_vals,
            coarse_bkg_weights,
            self.opt.importance_samples_per_ray,
            device=device
        )

        fine_bkg_out = self.net.fine_bkg_net(
            fine_bkg_pts,
            fine_bkg_dirs
        )
        fine_bkg_out = fine_bkg_out.detach()
        return fine_bkg_pts, fine_bkg_dirs, fine_bkg_z_vals, fine_bkg_out

    def _eval_human_samples(self, batch, device):
        human_batch = {
            'origin':    batch['origin'].clone().to(device),
            'direction': batch['direction'].clone().to(device),
            'near':      batch['human_near'].clone().to(device),
            'far':       batch['human_far'].clone().to(device),
        }
        human_samples = ray_utils.ray_to_samples(
            human_batch,
            self.opt.samples_per_ray,
            device=device,
            perturb=self.opt.perturb
        )
        human_pts = human_samples[0]
        human_dirs = human_samples[1]
        human_z_vals = human_samples[2]
        human_b, human_n, _ = human_pts.shape

        # predict offset
        cur_time = torch.ones_like(human_pts[..., 0:1]) * batch['cur_view_f'].to(device)
        offset = random.choice(self.net.offset_nets)(torch.cat([human_pts, cur_time], dim=-1))

        # warp points from observation space to canonical space
        mesh, raw_Ts = self.net.vertex_forward(int(batch['cap_id']))
        human_pts = human_pts.reshape(-1, 3)
        Ts, _, _ = ray_utils.warp_samples_to_canonical_diff(
            human_pts.detach().cpu().numpy(),
            verts=mesh[0],
            faces=self.val_dataset.scene.captures[batch['cap_id']].posed_mesh_cpu.faces_packed().numpy(),
            T=raw_Ts[0]
        )
        can_pts = (Ts @ ray_utils.to_homogeneous(human_pts)[..., None])[:, :3, 0]
        can_pts += offset.reshape(-1, 3)
        can_dirs = can_pts[1:] - can_pts[:-1]
        can_dirs = torch.cat([can_dirs, can_dirs[-1:]])
        can_dirs = can_dirs / torch.norm(can_dirs, dim=1, keepdim=True)
        can_pts = can_pts.reshape(human_b, human_n, 3)
        can_dirs = can_dirs.reshape(human_b, human_n, 3)
        human_out = self.net.coarse_human_net(can_pts, can_dirs)
        return human_pts, human_dirs, human_z_vals, can_pts, can_dirs, human_out

    def _color_range_regularization(self, pts, dirs, tgts):
        device = pts.device
        dummy_dirs = torch.randn(dirs.shape, dtype=dirs.dtype, device=device)
        dummy_dirs = dummy_dirs / torch.norm(dummy_dirs, dim=-1, keepdim=True)
        dummy_out = self.net.coarse_human_net(pts, dummy_dirs)
        color_reg = F.mse_loss(
            torch.sigmoid(dummy_out.reshape(-1, 4))[:, :3],
            torch.sigmoid(tgts.reshape(-1, 4))[:, :3]
        ) * self.penalize_color_range
        return color_reg

    def _smpl_symmetry_regularization(self, pts, dirs, tgts):
        '''
        we use dummy ray directions for the flipped points, since we only
        care about the occupancy symmetry. 
        '''
        pts_flip = pts.clone().detach()
        pts_flip[..., 0] *= -1
        out_flip = self.net.coarse_human_net(pts_flip, dirs.clone().detach())
        sym_reg = F.mse_loss(
            torch.tanh(torch.relu(tgts[..., 3])),
            torch.tanh(torch.relu(out_flip[..., 3]))
        ) * self.penalize_symmetric_alpha
        return sym_reg

    def _smpl_shape_regularization(self, batch, pts, dirs, pred):
        device = pts.device
        smpl_reg = torch.tensor(0.0, requires_grad=True).float().to(device)
        can_mesh = self.val_dataset.scene.captures[batch['cap_id']].can_mesh

        dist_human, _, _ = igl.signed_distance(
            pts.reshape(-1, 3).detach().cpu().numpy(),
            can_mesh.verts_packed().cpu().numpy(),
            can_mesh.faces_packed().cpu().numpy(),
        )
        inside_volume = dist_human < 0
        if inside_volume.sum() > 0:
            smpl_reg = smpl_reg + F.mse_loss(
                1 - torch.exp(-torch.relu(pred.reshape(-1, 4)[inside_volume][:, 3])),
                torch.ones_like(pred.reshape(-1, 4)[inside_volume][:, 3])
            ) * self.penalize_smpl_alpha

        # generate random samples inside a box in canonical space
        if self.penalize_dummy > 0:
            dummy_pts = (torch.rand(pts.shape, dtype=pts.dtype, device=device) - 0.5) * 3
            dummy_out = self.net.coarse_human_net(dummy_pts, dirs)
            dist_dummy, _, _ = igl.signed_distance(
                dummy_pts.reshape(-1, 3).detach().cpu().numpy(),
                can_mesh.verts_packed().cpu().numpy(),
                can_mesh.faces_packed().cpu().numpy(),
            )
            dummy_inside = dist_dummy < 0
            dummy_outside = dist_dummy > 0
            if dummy_inside.sum() > 0:
                smpl_reg = smpl_reg + F.mse_loss(
                    1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])),
                    torch.ones_like(dummy_out.reshape(-1, 4)[dummy_inside][:, 3])
                ) * self.penalize_dummy
            if dummy_outside.sum() > 0:
                smpl_reg = smpl_reg + F.l1_loss(
                    (1 - torch.exp(-torch.relu(dummy_out.reshape(-1, 4)[dummy_outside][:, 3]))) * torch.pow(torch.abs(torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor, self.opt.dist_exponent),
                    torch.zeros_like(dummy_out.reshape(-1, 4)[dummy_outside][:, 3])
                ) * self.penalize_dummy
        return smpl_reg

    def _sparsity_regularization(self, device):
        sparsity_reg = torch.tensor(0.0, requires_grad=True).float().to(device)
        # pick a random camera
        num_can_rays = 128
        can_cap = random.choice(self.can_caps)
        coords = np.argwhere(np.ones(can_cap.shape))
        coords = coords[np.random.randint(0, len(coords), num_can_rays)][:, ::-1]  # could get duplicated rays
        can_orig, can_dir = ray_utils.shot_rays(can_cap, coords)
        can_pts, can_dirs, can_z_vals = ray_utils.ray_to_samples(
            {
                'origin':    torch.from_numpy(can_orig).float().to(device),
                'direction': torch.from_numpy(can_dir).float().to(device),
                'near':      torch.zeros(num_can_rays, 1).float().to(device),
                'far':       torch.ones(num_can_rays, 1).float().to(device) * CANONICAL_CAMERA_DIST * 1.667,
            },
            samples_per_ray=self.opt.samples_per_ray,
            device=device,
            perturb=self.opt.perturb
        )
        can_out = self.net.coarse_human_net(can_pts, can_dirs)
        # compensate the interval difference between observation space and canonical space
        can_out[..., -1] *= self.interval_comp
        _, _, can_mask, can_weights, _ = render_utils.raw2outputs(can_out, can_z_vals.clone(), can_dirs[:, 0, :].clone(), white_bkg=True)
        can_weights = torch.clip(can_weights, 0.0, 1.0)
        can_mask = torch.clip(can_mask, 0.0, 1.0)
        # sharp edge loss
        if self.penalize_sharp_edge > 0:
            sparsity_reg = sparsity_reg + torch.mean(-torch.log(
                torch.exp(-torch.abs(can_mask)) + torch.exp(-torch.abs(1-can_mask))
            ) + HARD_SURFACE_OFFSET) * self.penalize_sharp_edge
        # hard surface loss
        if self.penalize_hard_surface > 0:
            sparsity_reg = sparsity_reg + torch.mean(-torch.log(
                torch.exp(-torch.abs(can_weights)) + torch.exp(-torch.abs(1-can_weights))
            ) + HARD_SURFACE_OFFSET) * self.penalize_hard_surface
        return sparsity_reg

    def loss_func(self, batch, return_rgb=False):
        device = next(self.net.parameters()).device
        loss_dict = {l: torch.tensor(0.0, requires_grad=True).float().to(device) for l in LOSS_NAMES}

        batch = utils.remove_first_axis(batch)
        hit_index = torch.nonzero(batch['is_hit'])[:, 0]
        _, fine_bkg_dirs, fine_bkg_z_vals, fine_bkg_out = self._eval_bkg_samples(batch, device)
        _, human_dirs, human_z_vals, can_pts, can_dirs, human_out = self._eval_human_samples(batch, device)

        # canonical space should be symmetric in terms of occupancy
        if self.penalize_symmetric_alpha > 0:
            loss_dict['smpl_sym_reg'] = loss_dict['smpl_sym_reg'] + self._smpl_symmetry_regularization(can_pts, can_dirs, human_out)

        # color of the same point should not change too much due to viewing directions
        if self.penalize_color_range > 0:
            loss_dict['color_range_reg'] = loss_dict['color_range_reg'] + self._color_range_regularization(can_pts, can_dirs, human_out)

        # the rendered human should be close to the detected human mask
        # loosely enforced, the penalty linearly decrease during training
        if self.penalize_mask > 0:
            _, _, human_mask, _, _ = render_utils.raw2outputs(human_out, human_z_vals, human_dirs[:, 0, :], white_bkg=self.opt.white_bkg)
            loss_dict['mask_loss'] = loss_dict['mask_loss'] + F.mse_loss(torch.clamp(human_mask, min=0.0, max=1.0), (1-batch['is_bkg']).float().to(device)) * self.penalize_mask

        # alpha inside smpl mesh should be 1
        # alpha outside smpl mesh should be 0
        if self.penalize_smpl_alpha > 0:
            loss_dict['smpl_shape_reg'] = loss_dict['smpl_shape_reg'] + self._smpl_shape_regularization(batch, can_pts, can_dirs, human_out)

        # sharp edge loss + hard surface loss
        if self.penalize_sharp_edge > 0 or self.penalize_hard_surface > 0:
            loss_dict['sparsity_reg'] = loss_dict['sparsity_reg'] + self._sparsity_regularization(device)

        # RGB loss
        fine_total_zvals, fine_order = torch.sort(torch.cat([fine_bkg_z_vals, human_z_vals], -1), -1)
        fine_total_out = torch.cat([fine_bkg_out, human_out], 1)
        _b, _n, _c = fine_total_out.shape
        fine_total_out = fine_total_out[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            fine_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]
        fine_rgb_map, _, _, _, _ = render_utils.raw2outputs(
            fine_total_out,
            fine_total_zvals,
            fine_bkg_dirs[:, 0, :],
            white_bkg=self.opt.white_bkg
        )
        loss_dict['fine_rgb_loss'] = loss_dict['fine_rgb_loss'] + F.mse_loss(fine_rgb_map[hit_index.to(device)], batch['color'][hit_index].to(device))

        # LPIPS loss
        if self.penalize_lpips > 0 and batch['patch_counter'] == 1:
            temp_lpips_loss = self.lpips_loss_fn(fine_rgb_map[:PATCH_SIZE_SQUARED].reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0, 1)*2-1, batch['color'][:PATCH_SIZE_SQUARED].to(device).reshape(PATCH_SIZE, PATCH_SIZE, -1).permute(2, 0, 1)*2-1) * self.penalize_lpips
            assert torch.numel(temp_lpips_loss) == 1
            loss_dict['lpips_loss'] = loss_dict['lpips_loss'] + temp_lpips_loss.flatten()[0]

        # restart if the network is dead
        if human_out[..., 3].max() <= 0.0:
            print('bad weights, reinitializing')
            self.net.offset_nets.apply(weight_reset)
            self.net.coarse_human_net.apply(weight_reset)
            loss_dict = {l: torch.tensor(0.0, requires_grad=True).float().to(device) for l in LOSS_NAMES}
        if return_rgb:
            return loss_dict, fine_rgb_map
        else:
            return loss_dict

    def validate_batch(self, batch):
        self.optim.zero_grad()
        assert self.net.training is False
        with torch.no_grad():
            loss_dict = self.loss_func(batch)
            loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
            loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']
        return {k: v.data.item() for k, v in loss_dict.items()}

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.net.training
        self.net.eval()
        all_loss = {l: [] for l in LOSS_NAMES}
        all_loss['rgb_loss'] = []
        all_loss['can_loss'] = []
        all_loss['total_loss'] = []
        for batch_idx, batch in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            cur_loss = self.validate_batch(batch)
            for k, v in cur_loss.items():
                all_loss[k].append(v)

        # draw visualizations
        overfit_id = self.train_dataset.scene.fname_to_index_dict[self.train_dataset.inclusions[len(self.train_dataset.inclusions) // 2]]
        overfit_cap = self.train_dataset.scene.captures[overfit_id]
        verts, _ = self.net.vertex_forward(overfit_id)
        verts = verts[0]
        faces = torch.from_numpy(self.val_dataset.scene.faces[:, :3]).to(verts.device)
        overlay = render_utils.overlay_smpl(overfit_cap.image, verts, faces, overfit_cap)

        if self.opt.random_view:
            can_cap = random.choice(self.can_caps)
        else:
            can_cap = self.can_caps[0]
        rgb_map, depth_map, acc_map = render_utils.render_smpl_nerf(
            self.net,
            can_cap,
            self.val_dataset.scene.static_vert[0],
            self.val_dataset.scene.faces,
            None,
            rays_per_batch=self.opt.rays_per_batch,
            samples_per_ray=self.opt.samples_per_ray,
            white_bkg=True,
            tpose=True,
            return_mask=True,
            return_depth=True,
            interval_comp=self.interval_comp
        )
        try:
            alpha_mask = acc_map >= 0.9999999999
            d_min = depth_map[alpha_mask].min()
            d_max = depth_map[alpha_mask].max()
            depth_map[depth_map <= d_min] = d_min
            depth_map[depth_map >= d_max] = d_max
        except:
            pass
        acc_map = np.stack([acc_map]*3, -1)
        depth_map = np.stack([depth_map]*3, -1)

        validation_data = {k: np.array(v).mean() for k, v in all_loss.items()}
        validation_data['render'] = utils.np_img_to_torch_img(np.stack([rgb_map, depth_map, acc_map, overlay]))
        self.push_validation_data(validation_data)
        self.save_model()
        if training:
            self.net.train()

    def save_model(self):
        save_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'hybrid_model_state_dict': self.net.state_dict(),
        }
        torch.save(save_dict, os.path.join(self.out, 'checkpoint.pth.tar'))

    def push_validation_data(self, validation_data):
        render = vutils.make_grid(validation_data['render'], nrow=2, normalize=True, scale_each=True)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        for key in validation_data.keys():
            if 'loss' in key or 'reg' in key:
                if key == 'lpips_loss' and validation_data[key] == 0:
                    pass
                else:
                    tb_datapack.add_scalar({f'val_loss/{key}': validation_data[key]})
        tb_datapack.add_image({'render/val': render})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def train_batch(self, batch):
        '''train for one batch of data
        '''
        self.optim.zero_grad()
        loss_dict, fine_rgb_map = self.loss_func(batch, return_rgb=True)
        loss_dict['rgb_loss'] = loss_dict['fine_rgb_loss'] + loss_dict['color_range_reg'] + loss_dict['lpips_loss']
        loss_dict['can_loss'] = loss_dict['smpl_sym_reg'] + loss_dict['smpl_shape_reg']
        if self.iteration >= self.opt.delay_iters:
            loss_dict['total_loss'] = loss_dict['rgb_loss'] + loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']
        else:
            loss_dict['total_loss'] = loss_dict['can_loss'] + loss_dict['mask_loss'] + loss_dict['sparsity_reg']
        losses = {k: v.data.item() for k, v in loss_dict.items()}

        if np.isnan(loss_dict['total_loss'].data.item()):
            print('loss is nan during training', losses)
            self.optim.zero_grad()
        else:
            loss_dict['total_loss'].backward()
            if self.opt.block_grad:
                try:
                    cap_id = int(batch['cap_id'].item())
                    grad_mask = turn_smpl_gradient_off(
                        self.train_dataset.scene.captures[cap_id].densepose
                    )
                    grad_mask = torch.from_numpy(grad_mask).float().to(
                        next(self.net.parameters()).device
                    )
                    self.net.poses.grad[cap_id] *= grad_mask
                except Exception as e:
                    print('failed to block gradients w.r.t unseen joints')
                    print(e)
            self.push_training_data(
                losses,
                self.optim.param_groups[0]['lr']
            )
        self.optim.step()

        # update learning rate
        if self.opt.lrate_decay is not None:
            decay_rate = 0.1
            decay_steps = self.opt.lrate_decay * 1000
            new_lrate = self.opt.learning_rate * (decay_rate ** (self.iteration / decay_steps))
            new_smpl_lrate = self.opt.smpl_lr * (decay_rate ** (self.iteration / decay_steps))
            for param_group in self.optim.param_groups[:1]:
                param_group['lr'] = new_smpl_lrate
            for param_group in self.optim.param_groups[1:3]:
                param_group['lr'] = new_lrate
            ###### update penalty ######
            # reduce prior knowledge based loss
            self.penalize_mask = self.opt.penalize_mask * max(0, 1 - (self.iteration / 60000))
            if self.opt.prior_knowledge_decay:
                self.penalize_symmetric_alpha = self.opt.penalize_symmetric_alpha * max(0, 1 - (self.iteration / 60000))
                self.penalize_dummy = self.opt.penalize_dummy * max(0, 1 - (self.iteration / 60000))
                self.penalize_smpl_alpha = self.opt.penalize_smpl_alpha * max(0, 1 - (self.iteration / 60000))

            assert self.opt.offset_lim >= self.opt.offset_scale >= 0
            new_offset_scale = ((self.opt.offset_lim - self.opt.offset_scale) * max(0, (self.iteration - self.opt.offset_delay) / 60000)) + self.opt.offset_scale
            for _offset_net in self.net.offset_nets:
                if self.iteration >= self.opt.offset_delay:
                    _offset_net.nerf.scale = min(new_offset_scale, self.opt.offset_lim)
                else:
                    _offset_net.nerf.scale = 0

    def train_epoch(self):
        '''train for one epoch
        one epoch is iterating the whole training dataset once
        '''
        self.net.train()
        for batch_idx, data_pack in tqdm.tqdm(enumerate(self.train_loader),
                                              initial=self.iteration % len(
                                                  self.train_loader),
                                              total=len(self.train_loader),
                                              desc='Train epoch={0}'.format(
                                                  self.epoch),
                                              ncols=80,
                                              leave=True,
                                              ):

            if self.iteration % self.valid_iter == 0:
                time.sleep(2)  # Prevent possible deadlock during epoch transition
                with torch.no_grad():
                    self.validate()

            self.train_batch(data_pack)

            if self.iteration >= self.max_iter:
                break
            self.iteration += 1

    def push_training_data(self, losses, lr):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        for key in losses.keys():
            if 'loss' in key or 'reg' in key:
                if key == 'lpips_loss' and losses[key] == 0:
                    pass
                else:
                    tb_datapack.add_scalar({f'train_loss/{key}': losses[key]})
        tb_datapack.add_scalar({'lr/lr': lr})
        tb_datapack.add_scalar({'hyper_params/offset_scale': self.net.offset_nets[0].nerf.scale})
        tb_datapack.add_scalar({'hyper_params/penalize_mask': self.penalize_mask})
        tb_datapack.add_scalar({'hyper_params/penalize_symmetric_alpha': self.penalize_symmetric_alpha})
        tb_datapack.add_scalar({'hyper_params/penalize_dummy': self.penalize_dummy})
        tb_datapack.add_scalar({'hyper_params/penalize_smpl_alpha': self.penalize_smpl_alpha})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.
        '''
        if hasattr(self.opt, 'load_weights'):
            assert self.opt.load_weights is None or self.opt.load_weights == False
        # 1. load check point
        checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(f'model check point cannot found: {checkpoint_path}')
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        utils.print_notification([f'Resuming from {self.iteration} iterations'])

    def load_pretrained_weights(self):
        '''
        load pretrained weights from another model
        '''
        # if hasattr(self.opt, 'resume'):
        #     assert self.opt.resume is False
        assert os.path.isfile(self.opt.load_weights_path), self.opt.load_weights_path
        content_list = []
        saved = torch.load(self.opt.load_weights_path, map_location='cpu')
        utils.safe_load_weights(self.net, saved['hybrid_model_state_dict'])
        content_list += [f'Loaded pretrained weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)
