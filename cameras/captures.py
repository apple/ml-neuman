# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/cameras/capture.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


'''
capture = pinhole camera + camera pose + captured content
'''


import numpy as np

from cameras.pinhole_camera import resize_pinhole_camera
from cameras import contents
from geometry import pcd_projector
from utils import ray_utils


################ Pinhole Capture ################


class BasePinholeCapture():
    def __init__(self, pinhole_cam, cam_pose):
        self.cam_pose = cam_pose
        self.pinhole_cam = pinhole_cam

    def __str__(self):
        string = f'pinhole camera: {self.pinhole_cam}\ncamera pose: {self.cam_pose}'
        return string

    @property
    def mvp_mat(self):
        '''
        model-view-projection matrix (naming from opengl)
        '''
        return np.matmul(self.pinhole_cam.intrinsic_matrix, self.cam_pose.extrinsic_matrix)

    @property
    def intrinsic_matrix(self):
        return self.pinhole_cam.intrinsic_matrix

    @property
    def extrinsic_matrix(self):
        return self.cam_pose.extrinsic_matrix

    @property
    def shape(self):
        return self.pinhole_cam.shape

    @property
    def size(self):
        return self.shape

    def camera_poly(self, size=1):
        _, dir_0 = ray_utils.shot_ray(self, 0, 0)
        _, dir_1 = ray_utils.shot_ray(self, self.shape[1], 0)
        _, dir_2 = ray_utils.shot_ray(self, self.shape[1], self.shape[0])
        _, dir_3 = ray_utils.shot_ray(self, 0, self.shape[0])
        orig = self.cam_pose.camera_center_in_world
        pt_0 = dir_0 * size + orig
        pt_1 = dir_1 * size + orig
        pt_2 = dir_2 * size + orig
        pt_3 = dir_3 * size + orig
        return orig, pt_0, pt_1, pt_2, pt_3


class RigPinholeCapture(BasePinholeCapture):
    def __init__(self, pinhole_cam, cam_pose, view_id, cam_id):
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose)
        self.view_id = view_id
        self.cam_id = cam_id

    def __str__(self):
        string = f'{super().__str__()}\nview id: {self.view_id}, camera id: {self.cam_id}'
        return string


class ResizedPinholeCapture(BasePinholeCapture):
    def __init__(self, pinhole_cam, cam_pose, tgt_size):
        pinhole_cam = resize_pinhole_camera(pinhole_cam, tgt_size)
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose)


class DepthPinholeCapture(BasePinholeCapture):
    def __init__(self, depth_path, pinhole_cam, cam_pose, crop_cam):
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose, crop_cam)
        self.captured_depth = contents.CapturedDepth(depth_path, crop_cam, self.pinhole_cam_before)

    def read_depth_to_ram(self):
        return self.captured_depth.read_depth_to_ram()

    @property
    def depth_path(self):
        return self.captured_depth.depth_path

    @property
    def depth_map(self):
        _depth = self.captured_depth.depth_map
        assert (_depth >= 0).all()
        return _depth

    @property
    def point_cloud_world(self):
        return self.get_point_cloud_world_from_depth(feat_map=None)

    def get_point_cloud_world_from_depth(self, feat_map=None):
        _pcd = pcd_projector.PointCloudProjectorNp.img_to_pcd_3d(self.depth_map, self.pinhole_cam.intrinsic_matrix, img=feat_map, cam2world=self.cam_pose.camera_to_world).astype(np.float32)
        return _pcd


class RGBPinholeCapture(BasePinholeCapture):
    def __init__(self, image_path, pinhole_cam, cam_pose):
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose)
        self.captured_image = contents.CapturedImage(image_path)

    def read_image_to_ram(self):
        return self.captured_image.read_image_to_ram()

    @property
    def image_path(self):
        return self.captured_image.image_path

    @property
    def image(self):
        _image = self.captured_image.image
        assert _image.shape[0:2] == self.pinhole_cam.shape, f'image does not match with camera model: image shape: {_image.shape}, pinhole camera: {self.pinhole_cam}'
        return _image


class RGBDPinholeCapture(RGBPinholeCapture, DepthPinholeCapture):
    def __init__(self, img_path, depth_path, pinhole_cam, cam_pose, crop_cam):
        RGBPinholeCapture.__init__(self, img_path, pinhole_cam, cam_pose, crop_cam)
        DepthPinholeCapture.__init__(self, depth_path, pinhole_cam, cam_pose, crop_cam)

    @property
    def point_cloud_w_rgb_world(self):
        return self.get_point_cloud_world_from_depth(feat_map=self.image)


class ResizedRGBPinholeCapture(ResizedPinholeCapture, RGBPinholeCapture):
    def __init__(self, image_path, pinhole_cam, cam_pose, tgt_size):
        ResizedPinholeCapture.__init__(self, pinhole_cam, cam_pose, tgt_size)
        self.captured_image = contents.ResizedCapturedImage(image_path, tgt_size)


class ResizedRGBDPinholeCapture(ResizedPinholeCapture, RGBDPinholeCapture):
    def __init__(self, image_path, depth_path, pinhole_cam, cam_pose, tgt_size):
        ResizedPinholeCapture.__init__(self, pinhole_cam, cam_pose, tgt_size)
        self.captured_image = contents.ResizedCapturedImage(image_path, tgt_size)
        self.captured_depth = contents.ResizedCapturedDepth(depth_path, tgt_size)


class RigRGBPinholeCapture(RigPinholeCapture, RGBPinholeCapture):
    def __init__(self, image_path, pinhole_cam, cam_pose, view_id, cam_id):
        RigPinholeCapture.__init__(self, pinhole_cam, cam_pose, view_id, cam_id)
        self.captured_image = contents.CapturedImage(image_path)


class RigRGBDPinholeCapture(RigPinholeCapture, RGBDPinholeCapture):
    def __init__(self, image_path, depth_path, pinhole_cam, cam_pose, view_id, cam_id):
        RigPinholeCapture.__init__(self, pinhole_cam, cam_pose, view_id, cam_id)
        self.captured_image = contents.CapturedImage(image_path)
        self.captured_depth = contents.CapturedDepth(depth_path)


class ResizedRigRGBPinholeCapture(RigPinholeCapture, ResizedRGBPinholeCapture):
    def __init__(self, image_path, pinhole_cam, cam_pose, tgt_size, view_id, cam_id):
        RigPinholeCapture.__init__(self, pinhole_cam, cam_pose, view_id, cam_id)
        ResizedRGBPinholeCapture.__init__(self, image_path, pinhole_cam, cam_pose, tgt_size)


class ResizedRigRGBDPinholeCapture(RigPinholeCapture, ResizedRGBDPinholeCapture):
    def __init__(self, image_path, depth_map, pinhole_cam, cam_pose, tgt_size, view_id, cam_id):
        RigPinholeCapture.__init__(self, pinhole_cam, cam_pose, view_id, cam_id)
        ResizedRGBDPinholeCapture.__init__(self, image_path, depth_map, pinhole_cam, cam_pose, tgt_size)
