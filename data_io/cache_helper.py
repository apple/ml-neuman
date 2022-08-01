#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
export cache for samples...
'''
import os

import numpy as np
import torch

from utils import ray_utils


def export_near_far_cache(opt, scene, geo_threshold, chunk, device):
    h, w = scene.captures[0].shape
    for cap in scene.captures:
        save_path = os.path.abspath(os.path.join(scene.captures[0].image_path, f'../../cache/near_far_cache_{os.path.basename(cap.image_path)}_{h}_{w}_{geo_threshold}_{opt.normalize}.npy'))

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if os.path.isfile(save_path):
            print(f'cache file already exists: {save_path}')
            continue
        near_far_cache = np.ones([h, w, 3])
        print('caching near/far', cap.image_path)
        coords = np.argwhere(np.ones_like(cap.mask) != 0)[:, ::-1]  # all pixel locations
        orig, dir = ray_utils.shot_rays(cap, coords)
        orig, dir = torch.from_numpy(orig).to(device), torch.from_numpy(dir).to(device)
        for k in range(0, coords.shape[0], chunk):
            temp_near, temp_far = ray_utils.geometry_guided_near_far(orig[k:k+chunk], dir[k:k+chunk], scene.verts[scene.image_path_to_index[cap.image_path]], geo_threshold)
            temp_near, temp_far = temp_near.detach().cpu().numpy(), temp_far.detach().cpu().numpy()
            near_far_cache[coords[k:k+chunk, 1], coords[k:k+chunk, 0]] = np.array([temp_near, temp_far, np.ones_like(temp_far)]).T
        np.save(save_path, near_far_cache)
        print(f'saved cache file to: {save_path}')


def load_near_far_cache(opt, scene, geo_threshold):
    cache_book = {}
    h, w = scene.captures[0].shape
    for cap in scene.captures:
        save_path = os.path.abspath(os.path.join(scene.captures[0].image_path, f'../../cache/near_far_cache_{os.path.basename(cap.image_path)}_{h}_{w}_{geo_threshold}_{opt.normalize}.npy'))
        assert os.path.isfile(save_path), f'{save_path} not exist'
        cur_cache = np.load(save_path)
        cache_book[os.path.basename(cap.image_path)] = cur_cache
        print(f'loaded near/far cache from: {save_path}')
    return cache_book
