#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import numpy as np
import torch
from torch.utils import data

from data_io import neuman_helper
from utils import ray_utils
from utils.constant import TRAIN_SET_LENGTH, VALIDATION_SET_LENGTH


class BackgroundRayDataset(data.Dataset):
    def __init__(self, opt, scene, dset_type, split):
        '''
        Args:
            opt (Namespace): options
            scene (BaseScene): the background scene
            dset_type (str): train/val/test set
            split (str): split file path
        '''
        self.opt = opt
        self.batch_size = opt.rays_per_batch
        self.scene = scene
        self.dset_type = dset_type
        self.split = split
        self.inclusions = neuman_helper.read_text(split)
        print(f'{dset_type} dataset has {len(self.inclusions)} samples: {self.inclusions}')

    def __len__(self):
        '''
        We use a large number to represent the length of the training set, and a small number for the validation set.
        '''
        if self.dset_type == 'train':
            return TRAIN_SET_LENGTH
        elif self.dset_type == 'val':
            return VALIDATION_SET_LENGTH
        else:
            raise ValueError

    def __getitem__(self, index):
        '''
        We sample random rays for each iteration inside the __getitem__ func, therefore returning tensor shape will be [1, num_rays, ...].
        Index is ignored, we randomly sample rays from all views.
        '''
        caps = [self.scene[fname] for fname in self.inclusions]
        bins = np.random.multinomial(self.batch_size, np.ones((len(caps))) / float(len(caps)))

        colors_list = []       # RGB colors
        depths_list = []       # MVS/fused depth values
        orig_list   = []       # Origins
        dir_list    = []       # Directions
        near_list   = []       # Near plans
        far_list    = []       # Far plans
        is_bkg_list = []       # Background/human mask
        viewf_list  = []       # Timestamps, for NeRF-T ablation
        for cap, num in zip(caps, bins):
            if num == 0:
                continue
            img = cap.image
            h, w, _ = img.shape
            if self.opt.ablate_nerft:
                # sample over the whole image
                # for NeRF-T ablation study use
                y = np.random.randint(0, h, (num))
                x = np.random.randint(0, w, (num))
                coords = np.stack([x, y], axis=1)
            elif hasattr(cap, 'border_mask'):
                # sample over the dilated background mask
                assert hasattr(cap, 'binary_mask')
                coords = np.argwhere((cap.border_mask | cap.mask) == 0)
                coords = coords[np.random.randint(0, len(coords), num)][:, ::-1]
            elif hasattr(cap, 'binary_mask'):
                # sample over the raw background mask
                coords = np.argwhere(cap.mask == 0)
                coords = coords[np.random.randint(0, len(coords), num)][:, ::-1]
            else:
                raise ValueError
            colors = (img[coords[:, 1], coords[:, 0]] / 255).astype(np.float32)
            if self.opt.use_fused_depth:
                depths = (cap.fused_depth_map[coords[:, 1], coords[:, 0]]).astype(np.float32)
            else:
                depths = (cap.depth_map[coords[:, 1], coords[:, 0]]).astype(np.float32)
            orig, dir = ray_utils.shot_rays(cap, coords)
            near = np.stack([[cap.near['bkg']]] * num)
            far = np.stack([[cap.far['bkg']]] * num)

            colors_list.append(colors)
            depths_list.append(depths)
            orig_list.append(orig)
            dir_list.append(dir)
            near_list.append(near)
            far_list.append(far)
            is_bkg_list.append(np.ones_like(far))
            viewf_list.append(np.ones_like(near) * cap.frame_id['frame_id'] / cap.frame_id['total_frames'])

        colors_list = np.concatenate(colors_list)
        depths_list = np.concatenate(depths_list)
        orig_list   = np.concatenate(orig_list)
        dir_list    = np.concatenate(dir_list)
        near_list   = np.concatenate(near_list)
        far_list    = np.concatenate(far_list)
        is_bkg_list = np.concatenate(is_bkg_list)
        viewf_list  = np.concatenate(viewf_list)
        assert colors_list.shape[0] == \
               depths_list.shape[0] == \
               orig_list.shape[0] == \
               dir_list.shape[0] == \
               near_list.shape[0] == \
               far_list.shape[0] ==\
               self.batch_size
        out = {
            'color':      torch.from_numpy(colors_list).float(),
            'depth':      torch.from_numpy(depths_list).float(),
            'origin':     torch.from_numpy(orig_list).float(),
            'direction':  torch.from_numpy(dir_list).float(),
            'near':       torch.from_numpy(near_list).float(),
            'far':        torch.from_numpy(far_list).float(),
            'is_bkg':     torch.from_numpy(is_bkg_list).long(),
            'viewf_list': torch.from_numpy(viewf_list).float(),
        }
        return out
