#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import copy

import torch
import torch.nn as nn

from utils import utils, ray_utils
from models import vanilla
from models.smpl import SMPL

'''
Extra offset network to compensate the misalignment
'''


class HumanNeRF(nn.Module):
    def __init__(self, opt, poses=None, betas=None, alignments=None, scale=None):
        super().__init__()
        self.coarse_bkg_net, self.fine_bkg_net = vanilla.build_nerf(opt)
        self.offset_nets = nn.ModuleList([vanilla.build_offset_net(opt) for i in range(opt.num_offset_nets)])
        # canonical space always use 0 as minimum frequency
        temp_opt = copy.deepcopy(opt)
        temp_opt.pos_min_freq = 0
        temp_opt.use_viewdirs = temp_opt.specular_can
        temp_opt.posenc = temp_opt.can_posenc
        self.coarse_human_net, _ = vanilla.build_nerf(temp_opt)
        if poses is not None:
            assert betas is not None
            assert alignments is not None
            assert scale is not None
            self.poses = torch.nn.parameter.Parameter(torch.from_numpy(poses).float(), requires_grad=True)
            self.betas = torch.nn.parameter.Parameter(torch.from_numpy(betas).float(), requires_grad=True)
            self.alignments = torch.nn.parameter.Parameter(torch.from_numpy(alignments).float(), requires_grad=True)
            self.scale = scale
            self.body_model = SMPL(
                os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl'),
                gender='neutral',
                device=torch.device('cpu')
            )
            self.da_smpl = torch.zeros_like(self.poses[0])
            self.da_smpl = self.da_smpl.reshape(-1, 3)
            self.da_smpl[1] = torch.tensor([0, 0, 1.0]).float()
            self.da_smpl[2] = torch.tensor([0, 0, -1.0]).float()
            self.da_smpl = torch.nn.parameter.Parameter(self.da_smpl.reshape(1, -1))

            self.poses_orig = poses.copy()
            self.betas_orig = betas.copy()

        try:
            pretrained_bkg = os.path.join(opt.out_dir, opt.load_background, 'checkpoint.pth.tar')
            bkg_weights = torch.load(pretrained_bkg, map_location='cpu')
            utils.safe_load_weights(self.coarse_bkg_net, bkg_weights['coarse_model_state_dict'])
            utils.safe_load_weights(self.fine_bkg_net, bkg_weights['fine_model_state_dict'])
            print(f'pretrained background model loaded from {pretrained_bkg}')
        except Exception as e:
            print(e)
            print('train from scratch')

        try:
            pretrained_can = os.path.join(opt.out_dir, opt.load_can, 'checkpoint.pth.tar')
            can_weights = torch.load(pretrained_can, map_location='cpu')
            _can_weights = {}
            for k in can_weights['hybrid_model_state_dict'].keys():
                if 'coarse_human_net.' in k:
                    _can_weights[k.split('coarse_human_net.', 1)[1]] = can_weights['hybrid_model_state_dict'][k]
            utils.safe_load_weights(self.coarse_human_net, _can_weights)
            print(f'pretrained canonical human model loaded from {pretrained_can}')
        except Exception as e:
            print(e)
            print('train from scratch')

        if opt.use_cuda:
            self.coarse_bkg_net = self.coarse_bkg_net.cuda()
            self.fine_bkg_net = self.fine_bkg_net.cuda()
            self.offset_nets = self.offset_nets.cuda()
            self.coarse_human_net = self.coarse_human_net.cuda()
            if poses is not None:
                self.poses = torch.nn.Parameter(torch.tensor(poses, device='cuda').float(), requires_grad=True)
                self.betas = torch.nn.Parameter(torch.tensor(betas, device='cuda').float(), requires_grad=True)
                self.alignments = torch.nn.Parameter(torch.tensor(alignments, device='cuda').float(), requires_grad=True)
                self.body_model = SMPL(
                    os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl'),
                    gender='neutral',
                    device='cuda'
                )
                self.da_smpl = torch.nn.Parameter(torch.tensor(self.da_smpl.detach().numpy(), device='cuda').float(), requires_grad=False)

    def vertex_forward(self, idx, pose=None, beta=None):
        if pose is None:
            pose = self.poses[idx][None]
        if beta is None:
            beta = self.betas[idx][None]
        _, T_t2pose = self.body_model.verts_transformations(
            return_tensor=True,
            poses=pose,
            betas=beta,
            transl=None,
        )
        _, T_t2da = self.body_model.verts_transformations(
            return_tensor=True,
            poses=self.da_smpl,
            betas=beta,
            transl=None,
        )
        T_da2pose = T_t2pose @ torch.inverse(T_t2da)
        T_da2scene = self.alignments[idx].T @ T_da2pose
        s = torch.eye(4).to(T_da2scene.device)
        s[:3, :3] *= self.scale
        T_da2scene = s @ T_da2scene
        da_pose_verts = self.body_model(
            return_tensor=True,
            return_joints=False,
            poses=self.da_smpl,
            betas=beta,
            transl=None,
        )
        world_verts = torch.einsum('bni, bi->bn', T_da2scene[0], ray_utils.to_homogeneous(da_pose_verts))[:, :3][None]
        return world_verts, T_da2scene
