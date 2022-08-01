#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import random

import numpy as np
import torch
from torch.utils import data

from data_io import cache_helper
from utils import ray_utils
from utils.constant import PATCH_SIZE, PATCH_SIZE_SQUARED, NEAR_INDEX, FAR_INDEX, TRAIN_SET_LENGTH, VALIDATION_SET_LENGTH
from data_io import neuman_helper


def get_left_upper_corner(img, pos, size=PATCH_SIZE):
    '''
    pos - [x, y]
    return the left upper corner of a patch centered(as centered as possible) at pos.
    '''
    h, w, _ = img.shape
    lu_y = int(pos[1] - size // 2)
    lu_x = int(pos[0] - size // 2)
    if lu_y < 0:
        lu_y -= lu_y
    if lu_x < 0:
        lu_x -= lu_x
    if lu_y + size > h:
        lu_y -= (lu_y + size) - (h)
    if lu_x + size > w:
        lu_x -= (lu_x + size) - (w)
    return np.array([lu_x, lu_y])


class HumanRayDataset(data.Dataset):
    '''
    We use random rays from a SINGLE time stamp per batch.
    '''

    def __init__(self, opt, scene, dset_type, split, near_far_cache=None):
        '''
        Args:
            opt (Namespace): options
            scene (BaseScene): the background scene
            dset_type (str): train/val/test set
            split (str): split file path
            near_far_cache (dict, optional): The SMPL mesh guided near and far for each pixels. Defaults to None.
                                             Key: file name of image.
                                             Value: ndarray with shape [h, w, 3].
        '''
        self.opt = opt
        self.scene = scene
        self.device = scene.captures[0].posed_mesh.device
        self.dset_type = dset_type
        self.split = split
        self.inclusions = neuman_helper.read_text(split)
        print(f'{dset_type} dataset has {len(self.inclusions)} samples: {self.inclusions}')
        self.white_bkg = opt.white_bkg
        self.batch_size = opt.rays_per_batch
        if near_far_cache is None:
            cache_helper.export_near_far_cache(opt, scene, opt.geo_threshold, opt.chunk, self.device)
            self.near_far_cache = cache_helper.load_near_far_cache(opt, scene, opt.geo_threshold)
        else:
            self.near_far_cache = near_far_cache
        self.num_patch = 1 if opt.penalize_lpips > 0 else 0
        self.cap_id = None

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

    def get_num_rays_dict(self, num):
        num_body_rays = int(round(num * self.opt.body_rays_ratio))
        num_border_rays = int(round(num * self.opt.border_rays_ratio)) if self.opt.dilation > 0 else 0
        num_bkg_rays = int(round(num * self.opt.bkg_rays_ratio))
        leftover = num - num_body_rays - num_border_rays - num_bkg_rays
        arr = np.array([num_body_rays, num_border_rays, num_bkg_rays])
        arr[arr.argmax()] += leftover
        num_body_rays, num_border_rays, num_bkg_rays = arr
        assert min(num_body_rays, num_border_rays, num_bkg_rays) >= 0, f'{min(num_body_rays, num_border_rays, num_bkg_rays)}'
        assert sum(arr) == num, f'{arr}, {leftover}'
        return {
            'num_body_rays': num_body_rays,
            'num_border_rays': num_border_rays,
            'num_bkg_rays': num_bkg_rays,
        }

    def get_num_rays_dict_patch(self, num):
        assert self.num_patch == 1
        assert num == PATCH_SIZE_SQUARED
        return {'num_patch_rays': num}

    def __getitem__(self, index):
        '''
        NeRF requires 4K+ rays per gradient decent, so we will return the ray batch directly.
        '''
        if self.cap_id is None:
            cap_id = self.scene.fname_to_index_dict[random.choice(self.inclusions)]
        else:
            cap_id = self.cap_id
        assert 0 <= cap_id < len(self.scene.captures)
        caps = self.scene.get_captures_by_view_id(cap_id)
        assert len(caps) == 1, 'one camera per one iteration'
        if self.num_patch == 0:
            # not sampling patch
            bins = [self.batch_size]
        elif self.num_patch == 1:
            # sampline patch
            # first part of the batch is the sampled patch
            # random rays for the leftover
            assert self.batch_size > PATCH_SIZE_SQUARED
            bins = [PATCH_SIZE_SQUARED, self.batch_size-PATCH_SIZE_SQUARED]
            caps = [caps[0], caps[0]]
        else:
            raise ValueError('only support 1 patch')
        if random.random() < self.opt.body_rays_ratio:
            need_patch = True
        else:
            need_patch = False
        patch_counter = 0
        assert len(caps) == len(bins), f'{len(caps)} != {len(bins)}'

        colors_list     = []
        orig_list       = []
        dir_list        = []
        human_near_list = []
        human_far_list  = []
        bkg_near_list   = []
        bkg_far_list    = []
        is_bkg_list     = []
        is_hit_list     = []
        coords_list     = {i: [] for i in range(len(caps))}

        for cam_id, (cap, num) in enumerate(zip(caps, bins)):
            if num == 0:
                continue
            img = cap.image
            if self.num_patch == 1 and need_patch and patch_counter == 0:
                num_rays_dict = self.get_num_rays_dict_patch(num)
                patch_counter += 1
            else:
                num_rays_dict = self.get_num_rays_dict(num)

            for ray_key, num_rays in num_rays_dict.items():
                if num_rays == 0:
                    continue
                if ray_key == 'num_body_rays':
                    coords = np.argwhere(cap.mask != 0)
                elif ray_key == 'num_border_rays':
                    coords = np.argwhere(cap.border_mask == 1)
                elif ray_key == 'num_bkg_rays':
                    coords = np.argwhere(cap.mask == 0)
                elif ray_key == 'num_patch_rays':
                    seed = random.choice(np.argwhere(cap.mask != 0))
                    seed = get_left_upper_corner(img, seed[::-1])[::-1]
                    bound = seed + PATCH_SIZE
                    # A hacky way to obtain the coords
                    # Assuming argwhere returns in order
                    temp = np.zeros_like(cap.mask)
                    assert temp[seed[0]:bound[0], seed[1]:bound[1]].shape == (PATCH_SIZE, PATCH_SIZE), 'wrong patch size'
                    temp[seed[0]:bound[0], seed[1]:bound[1]] = 1
                    coords = np.argwhere(temp == 1)
                    # An alternative which does not use argwhere
                    # y, x = np.meshgrid(
                    #                 np.linspace(seed[1], bound[1]-1, num=PATCH_SIZE),
                    #                 np.linspace(seed[0], bound[0]-1, num=PATCH_SIZE),
                    #         ).astype(int)
                    # coords = np.stack([x, y], -1).reshape(-1, 2)
                    check = (img[seed[0]:bound[0], seed[1]:bound[1]] / 255).astype(np.float32)
                else:
                    raise ValueError

                if ray_key == 'num_patch_rays':
                    coords = coords[:, ::-1]
                else:
                    coords = coords[np.random.randint(0, len(coords), num_rays)][:, ::-1]  # could get duplicated rays
                coords_list[cam_id].append(coords)
                colors = (img[coords[:, 1], coords[:, 0]] / 255).astype(np.float32)
                if ray_key == 'num_patch_rays':
                    assert (colors.reshape(PATCH_SIZE, PATCH_SIZE, -1) == check).all(), 'rays not forming a patch'
                is_bkg = 1 - cap.binary_mask[coords[:, 1], coords[:, 0]]

                orig, dir = ray_utils.shot_rays(cap, coords)
                cache = self.near_far_cache[os.path.basename(cap.image_path)][coords[:, 1], coords[:, 0]]
                valid = cache[..., NEAR_INDEX] < cache[..., FAR_INDEX]
                human_near = np.stack([[cap.near['human']]] * num_rays)
                human_far = np.stack([[cap.far['human']]] * num_rays)
                human_near[valid, 0] = cache[valid][:, NEAR_INDEX]
                human_far[valid, 0] = cache[valid][:, FAR_INDEX]
                bkg_near = np.stack([[cap.near['bkg']]] * num_rays)
                bkg_far = np.stack([[cap.far['bkg']]] * num_rays)

                assert ((human_near <= human_far).all())
                colors_list.append(colors)
                orig_list.append(orig)
                dir_list.append(dir)
                human_near_list.append(human_near)
                human_far_list.append(human_far)
                bkg_near_list.append(bkg_near)
                bkg_far_list.append(bkg_far)
                is_bkg_list.append(is_bkg)
                is_hit_list.append(valid.astype(np.uint8))

        colors_list     = np.concatenate(colors_list)
        orig_list       = np.concatenate(orig_list)
        dir_list        = np.concatenate(dir_list)
        human_near_list = np.concatenate(human_near_list)
        human_far_list  = np.concatenate(human_far_list)
        bkg_near_list   = np.concatenate(bkg_near_list)
        bkg_far_list    = np.concatenate(bkg_far_list)
        is_bkg_list     = np.concatenate(is_bkg_list)
        is_hit_list     = np.concatenate(is_hit_list)
        assert len(coords_list) == len(bins), f'{len(coords_list)} != {len(bins)}'
        assert colors_list.shape[0] == \
               orig_list.shape[0] == \
               dir_list.shape[0] == \
               human_near_list.shape[0] == \
               human_far_list.shape[0] == \
               bkg_near_list.shape[0] == \
               bkg_far_list.shape[0] == \
               is_bkg_list.shape[0] == \
               is_hit_list.shape[0] == \
               self.batch_size
        out = {
            'color':         torch.from_numpy(colors_list).float(),
            'origin':        torch.from_numpy(orig_list).float(),
            'direction':     torch.from_numpy(dir_list).float(),
            'human_near':    torch.from_numpy(human_near_list).float(),
            'human_far':     torch.from_numpy(human_far_list).float(),
            'bkg_near':      torch.from_numpy(bkg_near_list).float(),
            'bkg_far':       torch.from_numpy(bkg_far_list).float(),
            'is_bkg':        torch.from_numpy(is_bkg_list).long(),
            'is_hit':        torch.from_numpy(is_hit_list).long(),
            'cur_view_f':    cap.frame_id['frame_id'] / cap.frame_id['total_frames'],
            'cur_view':      cap.frame_id['frame_id'],
            'cap_id':        cap_id,
            'patch_counter': torch.tensor(patch_counter),
        }
        return out
