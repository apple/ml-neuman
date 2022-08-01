# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/projector/pcd_projector.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


'''
a point cloud projector based on numpy
'''

import numpy as np
import torch


def transform_points(points, transformation):
    append_ones = np.ones_like(points[:, 0:1])
    xyzw = np.concatenate([points, append_ones], axis=1)
    xyzw = np.matmul(transformation, xyzw.T).T
    xyzw /= xyzw[:, 3:4]
    points = xyzw[:, 0:3]
    return points


def project_point_cloud_at_capture(point_cloud, capture, render_type='rgb'):
    if render_type == 'rgb':
        assert point_cloud.shape[1] == 6
    else:
        point_cloud = point_cloud[:, :3]
        assert point_cloud.shape[1] == 3
    if render_type in ['bw', 'rgb']:
        keep_z = False
    else:
        keep_z = True

    pcd_2d = PointCloudProjectorNp.pcd_3d_to_pcd_2d(point_cloud,
                                                    capture.intrinsic_matrix,
                                                    capture.extrinsic_matrix,
                                                    capture.size,
                                                    keep_z=True,
                                                    crop=True,
                                                    filter_neg=True,
                                                    norm_coord=False,
                                                    return_index=False)
    if render_type == 'pcd':
        return pcd_2d
    reproj = PointCloudProjectorNp.pcd_2d_to_img(pcd_2d,
                                                 capture.size,
                                                 has_z=True,
                                                 keep_z=keep_z)
    return reproj


def pcd_3d_to_pcd_2d_torch(pcd, intrinsic, extrinsic, size, keep_z, crop: bool = True, filter_neg: bool = True, norm_coord: bool = True, return_index: bool = False, valid_mask=None):
    assert isinstance(pcd, torch.Tensor), f'cannot process data type: {type(pcd)}'
    assert isinstance(intrinsic, torch.Tensor), f'cannot process data type: {type(intrinsic)}'
    assert isinstance(extrinsic, torch.Tensor), f'cannot process data type: {type(extrinsic)}'
    assert len(pcd.shape) == 3 and pcd.shape[1] >= 3 and pcd.shape[2] > pcd.shape[1], f'seems the input pcd is not a valid point cloud: {pcd.shape}'
    assert intrinsic.shape[1:] == (3, 3)
    if extrinsic.shape[1:] == (4, 4):
        extrinsic = extrinsic[:, 0:3, :]
    assert extrinsic.shape[1:] == (3, 4)

    if valid_mask is None:
        valid_mask = pcd.bool().all(dim=1)
    xyzw = torch.cat([pcd[:, 0:3, :], torch.ones_like(pcd[:, 0:1, :])], axis=1)
    mvp_mat = torch.matmul(intrinsic, extrinsic)
    camera_points = torch.matmul(mvp_mat, xyzw)
    if filter_neg:
        valid_mask = valid_mask * (camera_points[:, 2, :] > 0.0)
    camera_points = valid_mask[:, None, :].float() * camera_points
    image_points = torch.zeros_like(camera_points)
    image_points = camera_points / camera_points[:, 2:3, :]
    image_points = image_points[:, :2, :]
    # clear nan
    image_points[torch.isnan(image_points).any(dim=1)[:, None, :].repeat(1, 2, 1)] = 0
    size = size.float()
    if crop:
        valid_mask = valid_mask * (image_points[:, 0, :] >= 0) * (image_points[:, 0, :] < (size[:, 1] - 1)[:, None]) * (image_points[:, 0, :] >= 0) * (image_points[:, 1, :] < (size[:, 0] - 1)[:, None])
    if norm_coord:
        image_points = ((image_points / torch.flip(size, [1])[..., None]) * 2) - 1
    if keep_z:
        image_points = torch.cat([image_points, camera_points[:, 2:3, :], pcd[:, 3:, :]], dim=1)
    else:
        image_points = torch.cat([image_points, pcd[:, 3:, :]], dim=1)
    image_points = valid_mask[:, None, :].float() * image_points
    if return_index:
        return image_points, valid_mask
    return image_points


class PointCloudProjectorNp():
    def __init__(self):
        pass

    @staticmethod
    def pcd_2d_to_pcd_3d(pcd, depth, intrinsic, cam2world=None):
        assert isinstance(pcd, np.ndarray), f'cannot process data type: {type(pcd)}'
        assert isinstance(intrinsic, np.ndarray), f'cannot process data type: {type(intrinsic)}'
        assert len(pcd.shape) == 2 and pcd.shape[1] >= 2
        assert len(depth.shape) == 2 and depth.shape[1] == 1
        assert intrinsic.shape == (3, 3)
        if cam2world is not None:
            assert isinstance(cam2world, np.ndarray), f'cannot process data type: {type(cam2world)}'
            assert cam2world.shape == (4, 4)

        x, y, z = pcd[:, 0], pcd[:, 1], depth[:, 0]
        append_ones = np.ones_like(x)
        xyz = np.stack([x, y, append_ones], axis=1)  # shape: [num_points, 3]
        inv_intrinsic_mat = np.linalg.inv(intrinsic)
        xyz = np.matmul(inv_intrinsic_mat, xyz.T).T * z[..., None]
        valid_mask_1 = np.where(xyz[:, 2] > 0)
        xyz = xyz[valid_mask_1]

        if cam2world is not None:
            append_ones = np.ones_like(xyz[:, 0:1])
            xyzw = np.concatenate([xyz, append_ones], axis=1)
            xyzw = np.matmul(cam2world, xyzw.T).T
            valid_mask_2 = np.where(xyzw[:, 3] != 0)
            xyzw = xyzw[valid_mask_2]
            xyzw /= xyzw[:, 3:4]
            xyz = xyzw[:, 0:3]

        if pcd.shape[1] > 2:
            features = pcd[:, 2:]
            try:
                features = features[valid_mask_1][valid_mask_2]
            except UnboundLocalError:
                features = features[valid_mask_1]
            assert xyz.shape[0] == features.shape[0]
            xyz = np.concatenate([xyz, features], axis=1)
        return xyz

    @staticmethod
    def img_to_pcd_3d(depth, intrinsic, img=None, cam2world=None):
        '''
        the function signature is not fully correct, because img is an optional
        if cam2world is None, the output pcd is in camera space, else the out pcd is in world space.
        here the output is pure np array
        '''

        assert isinstance(depth, np.ndarray), f'cannot process data type: {type(depth)}'
        assert len(depth.shape) == 2
        assert isinstance(intrinsic, np.ndarray), f'cannot process data type: {type(intrinsic)}'
        assert intrinsic.shape == (3, 3)
        if img is not None:
            assert isinstance(img, np.ndarray), f'cannot process data type: {type(img)}'
            assert len(img.shape) == 3
            assert img.shape[:2] == depth.shape[:2], 'feature should have the same resolution as the depth'
        if cam2world is not None:
            assert isinstance(cam2world, np.ndarray), f'cannot process data type: {type(cam2world)}'
            assert cam2world.shape == (4, 4)

        pcd_image_space = PointCloudProjectorNp.img_to_pcd_2d(depth[..., None], norm_coord=False)
        valid_mask_1 = np.where(pcd_image_space[:, 2] > 0)
        pcd_image_space = pcd_image_space[valid_mask_1]
        xy = pcd_image_space[:, :2]
        z = pcd_image_space[:, 2:3]
        if img is not None:
            _c = img.shape[-1]
            feat = img.reshape(-1, _c)
            feat = feat[valid_mask_1]
            xy = np.concatenate([xy, feat], axis=1)
        pcd_3d = PointCloudProjectorNp.pcd_2d_to_pcd_3d(xy, z, intrinsic, cam2world=cam2world)
        return pcd_3d

    @staticmethod
    def pcd_3d_to_pcd_2d(pcd, intrinsic, extrinsic, size, keep_z, crop=True, filter_neg=True, norm_coord=True, return_index=False):
        assert isinstance(pcd, np.ndarray), f'cannot process data type: {type(pcd)}'
        assert isinstance(intrinsic, np.ndarray), f'cannot process data type: {type(intrinsic)}'
        assert isinstance(extrinsic, np.ndarray), f'cannot process data type: {type(extrinsic)}'
        assert len(pcd.shape) == 2 and pcd.shape[1] >= 3, f'seems the input pcd is not a valid 3d point cloud: {pcd.shape}'

        xyzw = np.concatenate([pcd[:, 0:3], np.ones_like(pcd[:, 0:1])], axis=1)
        mvp_mat = np.matmul(intrinsic, extrinsic)
        camera_points = np.matmul(mvp_mat, xyzw.T).T
        if filter_neg:
            valid_mask_1 = camera_points[:, 2] > 0.0
        else:
            valid_mask_1 = np.ones_like(camera_points[:, 2], dtype=bool)
        camera_points = camera_points[valid_mask_1]
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        if crop:
            valid_mask_2 = (image_points[:, 0] >= 0) * (image_points[:, 0] < size[1] - 1) * (image_points[:, 1] >= 0) * (image_points[:, 1] < size[0] - 1)
        else:
            valid_mask_2 = np.ones_like(image_points[:, 0], dtype=bool)
        if norm_coord:
            image_points = ((image_points / size[::-1]) * 2) - 1

        if keep_z:
            image_points = np.concatenate([image_points[valid_mask_2], camera_points[valid_mask_2][:, 2:3], pcd[valid_mask_1][:, 3:][valid_mask_2]], axis=1)
        else:
            image_points = np.concatenate([image_points[valid_mask_2], pcd[valid_mask_1][:, 3:][valid_mask_2]], axis=1)
        if return_index:
            points_index = np.arange(pcd.shape[0])[valid_mask_1][valid_mask_2]
            return image_points, points_index
        return image_points

    @staticmethod
    def pcd_2d_to_img(pcd, size, has_z=False, keep_z=False):
        assert len(pcd.shape) == 2 and pcd.shape[-1] >= 2, f'seems the input pcd is not a valid point cloud: {pcd.shape}'
        if has_z:
            pcd = pcd[pcd[:, 2].argsort()[::-1]]
            if not keep_z:
                pcd = np.delete(pcd, [2], axis=1)
        index_list = np.round(pcd[:, 0:2]).astype(np.int32)
        index_list[:, 0] = np.clip(index_list[:, 0], 0, size[1] - 1)
        index_list[:, 1] = np.clip(index_list[:, 1], 0, size[0] - 1)
        _h, _w, _c = *size, pcd.shape[-1] - 2
        if _c == 0:
            canvas = np.zeros((_h, _w, 1))
            canvas[index_list[:, 1], index_list[:, 0]] = 1.0
        else:
            canvas = np.zeros((_h, _w, _c))
            canvas[index_list[:, 1], index_list[:, 0]] = pcd[:, 2:]

        return canvas

    @staticmethod
    def img_to_pcd_2d(img, norm_coord=True):
        assert isinstance(img, np.ndarray), f'cannot process data type: {type(img)}'
        assert len(img.shape) == 3

        _h, _w, _c = img.shape
        if norm_coord:
            x, y = np.meshgrid(
                np.linspace(-1, 1, num=_w),
                np.linspace(-1, 1, num=_h),
            )
        else:
            x, y = np.meshgrid(
                np.linspace(0, _w - 1, num=_w),
                np.linspace(0, _h - 1, num=_h),
            )
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        feat = img.reshape(-1, _c)
        pcd_2d = np.concatenate([x, y, feat], axis=1)
        return pcd_2d
