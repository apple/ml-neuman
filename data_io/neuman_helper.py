#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import joblib

import numpy as np
import torch
from tqdm import tqdm
import PIL
import scipy

from data_io import colmap_helper
from geometry import pcd_projector
from cameras import captures as captures_module, contents
from scenes import scene as scene_module
from utils import utils, ray_utils
from models.smpl import SMPL


class NeuManCapture(captures_module.RigRGBDPinholeCapture):
    def __init__(self, image_path, depth_path, mask_path, pinhole_cam, cam_pose, view_id, cam_id, mono_depth_path=None, keypoints_path=None, densepose_path=None):
        captures_module.RigRGBDPinholeCapture.__init__(self, image_path, depth_path, pinhole_cam, cam_pose, view_id, cam_id)
        self.captured_mask = contents.CapturedImage(
            mask_path
        )
        if mono_depth_path is not None:
            self.captured_mono_depth = contents.CapturedDepth(
                mono_depth_path
            )
            self.captured_mono_depth.dataset = 'mono'
        else:
            self.captured_mono_depth = None

        if keypoints_path is not None:
            self.keypoints = np.load(keypoints_path)
        else:
            self.keypoints = None

        if densepose_path is not None:
            self.densepose = np.load(densepose_path)
        else:
            self.densepose = None

        self._fused_depth_map = None

    def read_image_to_ram(self):
        if self.captured_mono_depth is None:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()
        else:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram() + self.captured_mono_depth.read_depth_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image.copy()
        if _mask.max() == 255:
            # Detectron2 mask
            _mask[_mask == 255] = 1
            _mask = 1 - _mask
        else:
            raise ValueError
        assert _mask.sum() > 0
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask

    @property
    def mono_depth_map(self):
        return self.captured_mono_depth.depth_map

    @property
    def fused_depth_map(self):
        if self._fused_depth_map is None:
            valid_mask = (self.depth_map > 0) & (self.mask == 0)
            x = self.mono_depth_map[valid_mask]
            y = self.depth_map[valid_mask]
            res = scipy.stats.linregress(x, y)
            self._fused_depth_map = self.depth_map.copy()
            self._fused_depth_map[~valid_mask] = self.mono_depth_map[~valid_mask] * res.slope + res.intercept
        return self._fused_depth_map


class ResizedNeuManCapture(captures_module.ResizedRigRGBDPinholeCapture):
    def __init__(self, image_path, depth_path, mask_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id, mono_depth_path=None, keypoints_path=None, densepose_path=None):
        captures_module.ResizedRigRGBDPinholeCapture.__init__(self, image_path, depth_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id)
        '''
        Note: we pass in the original intrinsic and distortion matrix, NOT the resized intrinsic
        '''
        self.captured_mask = contents.ResizedCapturedImage(
            mask_path,
            tgt_size,
            sampling=PIL.Image.NEAREST
        )
        if mono_depth_path is not None:
            self.captured_mono_depth = contents.ResizedCapturedDepth(
                mono_depth_path,
                tgt_size=tgt_size
            )
            self.captured_mono_depth.dataset = 'mono'
        else:
            self.captured_mono_depth = None
        if keypoints_path is not None:
            # raise NotImplementedError
            self.keypoints = None
        else:
            self.keypoints = None
        if densepose_path is not None:
            # raise NotImplementedError
            self.densepose = None
        else:
            self.densepose = None

    def read_image_to_ram(self):
        if self.captured_mono_depth is None:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram()
        else:
            return self.captured_image.read_image_to_ram() + self.captured_mask.read_image_to_ram() + self.captured_mono_depth.read_depth_to_ram()

    @property
    def mask(self):
        _mask = self.captured_mask.image.copy()
        if _mask.max() == 255:
            # Detectron2 mask
            _mask[_mask == 255] = 1
            _mask = 1 - _mask
        else:
            raise ValueError
        assert _mask.sum() > 0
        assert _mask.shape[0:2] == self.pinhole_cam.shape, f'mask does not match with camera model: mask shape: {_mask.shape}, pinhole camera: {self.pinhole_cam}'
        return _mask

    @property
    def binary_mask(self):
        _mask = self.mask.copy()
        _mask[_mask > 0] = 1
        return _mask

    @property
    def mono_depth_map(self):
        return self.captured_mono_depth.depth_map


def create_split_files(scene_dir):
    # 10% as test set
    # 10% as validation set
    # 80% as training set
    dummy_scene = NeuManReader.read_scene(scene_dir)
    scene_length = len(dummy_scene.captures)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0
    splits = []
    for l, split in zip([train_list, val_list, test_list], ['train', 'val', 'test']):
        output = []
        save_path = os.path.join(scene_dir, f'{split}_split.txt')
        for i, cap in enumerate(dummy_scene.captures):
            if i in l:
                output.append(os.path.basename(cap.image_path))
        with open(save_path, 'w') as f:
            for item in output:
                f.write("%s\n" % item)
        splits.append(save_path)
    return splits


def read_text(txt_file):
    '''
    read the split file to a list
    '''
    assert os.path.isfile(txt_file)
    items = []
    with open(txt_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            items.append(line.strip())
    return items


class NeuManReader():
    def __init__(self):
        pass

    @classmethod
    def read_scene(cls, scene_dir, tgt_size=None, normalize=False, bkg_range_scale=1.1, human_range_scale=1.1, mask_dir='segmentations', smpl_type='romp', keypoints_dir='keypoints', densepose_dir='densepose'):
        def update_near_far(scene, keys, range_scale):
            # compute the near and far
            for view_id in tqdm(range(scene.num_views), total=scene.num_views, desc=f'Computing near/far for {keys}'):
                for cam_id in range(scene.num_cams):
                    cur_cap = scene.get_capture_by_view_cam_id(view_id, cam_id)
                    if not hasattr(cur_cap, 'near'):
                        cur_cap.near = {}
                    if not hasattr(cur_cap, 'far'):
                        cur_cap.far = {}
                    for k in keys:
                        if k == 'bkg':
                            pcd_2d_bkg = pcd_projector.project_point_cloud_at_capture(scene.point_cloud, cur_cap, render_type='pcd')
                            near = 0  # np.percentile(pcd_2d_bkg[:, 2], 5)
                            far = np.percentile(pcd_2d_bkg[:, 2], 95)
                        elif k == 'human':
                            pcd_2d_human = pcd_projector.project_point_cloud_at_capture(scene.verts[view_id], cur_cap, render_type='pcd')
                            near = pcd_2d_human[:, 2].min()
                            far = pcd_2d_human[:, 2].max()
                        else:
                            raise ValueError(k)
                        center = (near + far) / 2
                        length = (far - near) * range_scale
                        cur_cap.near[k] = max(0.0, float(center - length / 2))
                        cur_cap.far[k] = float(center + length / 2)
        captures, point_cloud, num_views, num_cams = cls.read_captures(scene_dir, tgt_size, mask_dir=mask_dir, keypoints_dir=keypoints_dir, densepose_dir=densepose_dir)
        scene = scene_module.RigCameraScene(captures, num_views, num_cams)
        scene.point_cloud = point_cloud
        update_near_far(scene, ['bkg'], bkg_range_scale)

        if normalize:
            fars = []
            for cap in scene.captures:
                fars.append(cap.far['bkg'])
            fars = np.array(fars)
            scale = 3.14 / (np.percentile(fars, 95))
            for cap in scene.captures:
                cap.cam_pose.camera_center_in_world *= scale
                cap.near['bkg'], cap.far['bkg'] = cap.near['bkg'] * scale, cap.far['bkg'] * scale
                cap.captured_depth.scale = scale
                cap.captured_mono_depth.scale = scale
            scene.point_cloud[:, :3] *= scale
        else:
            scale = 1

        scene.scale = scale
        smpls, world_verts, static_verts, Ts = cls.read_smpls(scene_dir, scene.captures, scale=scale, smpl_type=smpl_type)
        scene.smpls, scene.verts, scene.static_vert, scene.Ts = smpls, world_verts, static_verts, Ts
        _, uvs, faces = utils.read_obj(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl_uv.obj')
        )
        scene.uvs, scene.faces = uvs, faces
        update_near_far(scene, ['human'], human_range_scale)

        assert len(scene.captures) > 0

        return scene

    @classmethod
    def read_smpls(cls, scene_dir, caps, scale=1, smpl_type='romp'):
        def extract_smpl_at_frame(raw_smpl, frame_id):
            out = {}
            for k, v in raw_smpl.items():
                try:
                    out[k] = v[frame_id]
                except:
                    out[k] = None
            return out

        device = torch.device('cpu')
        body_model = SMPL(
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl'),
            gender='neutral',
            device=device
        )
        smpls = []
        static_verts = []
        world_verts = []
        Ts = []
        smpl_path = os.path.join(scene_dir, f'smpl_output_{smpl_type}.pkl')
        assert os.path.isfile(smpl_path), f'{smpl_path} is missing'
        print(f'using {smpl_type} smpl')
        raw_smpl = joblib.load(smpl_path)
        assert len(raw_smpl) == 1
        raw_smpl = raw_smpl[list(raw_smpl.keys())[0]]
        raw_alignments = np.load(os.path.join(scene_dir, 'alignments.npy'), allow_pickle=True).item()
        for cap in caps:
            frame_id = int(os.path.basename(cap.image_path)[:-4])
            # assert 0 <= frame_id < len(caps)
            temp_smpl = extract_smpl_at_frame(raw_smpl, frame_id)
            temp_alignment = np.eye(4)
            temp_alignment[:, :3] = raw_alignments[os.path.basename(cap.image_path)]

            # 大 pose
            da_smpl = np.zeros_like(temp_smpl['pose'][None])
            da_smpl = da_smpl.reshape(-1, 3)
            da_smpl[1] = np.array([0, 0, 1.0])
            da_smpl[2] = np.array([0, 0, -1.0])
            da_smpl = da_smpl.reshape(1, -1)

            _, T_t2pose = body_model.verts_transformations(
                return_tensor=False,
                poses=temp_smpl['pose'][None],
                betas=temp_smpl['betas'][None],
                concat_joints=True
            )
            _, T_t2da = body_model.verts_transformations(
                return_tensor=False,
                poses=da_smpl,
                betas=temp_smpl['betas'][None],
                concat_joints=True
            )
            T_da2pose = T_t2pose @ np.linalg.inv(T_t2da)
            T_da2scene = temp_alignment.T @ T_da2pose
            s = np.eye(4)
            s[:3, :3] *= scale
            T_da2scene = s @ T_da2scene

            da_pose_verts, da_pose_joints = body_model(
                return_tensor=False,
                return_joints=True,
                poses=da_smpl,
                betas=temp_smpl['betas'][None]
            )
            temp_world_verts = np.einsum('BNi, Bi->BN', T_da2scene, ray_utils.to_homogeneous(np.concatenate([da_pose_verts, da_pose_joints], axis=0)))[:, :3].astype(np.float32)
            temp_world_verts, temp_world_joints = temp_world_verts[:6890, :], temp_world_verts[6890:, :]
            temp_smpl['joints_3d'] = temp_world_joints
            temp_smpl['static_joints_3d'] = da_pose_joints
            smpls.append(temp_smpl)
            Ts.append(T_da2scene)
            static_verts.append(da_pose_verts)
            world_verts.append(temp_world_verts)
        return smpls, world_verts, static_verts, Ts

    @classmethod
    def read_captures(cls, scene_dir, tgt_size, mask_dir='segmentations', keypoints_dir='keypoints', densepose_dir='densepose'):
        caps = []
        raw_scene = colmap_helper.ColmapAsciiReader.read_scene(
            os.path.join(scene_dir, 'sparse'),
            os.path.join(scene_dir, 'images'),
            tgt_size,
            order='video',
        )
        num_views = len(raw_scene.captures)
        num_cams = 1
        counter = 0
        for view_id in range(num_views):
            for cam_id in range(num_cams):
                raw_cap = raw_scene.captures[counter]
                depth_path = raw_cap.image_path.replace('/images/', '/depth_maps/') + '.geometric.bin'
                mono_depth_path = raw_cap.image_path.replace('/images/', '/mono_depth/')
                if not os.path.isfile(depth_path):
                    depth_path = raw_cap.image_path + 'dummy'
                    print(f'can not find mvs depth for {os.path.basename(raw_cap.image_path)}')
                if not os.path.isfile(mono_depth_path):
                    mono_depth_path = raw_cap.image_path + 'dummy'
                    print(f'can not find mono depth for {os.path.basename(raw_cap.image_path)}')
                mask_path = os.path.join(scene_dir, mask_dir, os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(mask_path):
                    mask_path = os.path.join(scene_dir, mask_dir, os.path.basename(raw_cap.image_path))
                keypoints_path = os.path.join(scene_dir, keypoints_dir, os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(keypoints_path):
                    print(f'can not find keypoints for {os.path.basename(raw_cap.image_path)}')
                    keypoints_path = None
                densepose_path = os.path.join(scene_dir, densepose_dir, 'dp_' + os.path.basename(raw_cap.image_path) + '.npy')
                if not os.path.isfile(densepose_path):
                    print(f'can not find densepose for {os.path.basename(raw_cap.image_path)}')
                    densepose_path = None
                if tgt_size is None:
                    temp = NeuManCapture(
                        raw_cap.image_path,
                        depth_path,
                        mask_path,
                        raw_cap.pinhole_cam,
                        raw_cap.cam_pose,
                        view_id,
                        cam_id,
                        mono_depth_path=mono_depth_path,
                        keypoints_path=keypoints_path,
                        densepose_path=densepose_path
                    )
                else:
                    temp = ResizedNeuManCapture(
                        raw_cap.image_path,
                        depth_path,
                        mask_path,
                        raw_cap.pinhole_cam,
                        raw_cap.cam_pose,
                        tgt_size,
                        view_id,
                        cam_id,
                        mono_depth_path=mono_depth_path,
                        keypoints_path=keypoints_path,
                        densepose_path=densepose_path
                    )
                temp.frame_id = raw_cap.frame_id
                counter += 1
                caps.append(temp)
        return caps, raw_scene.point_cloud, num_views, num_cams
