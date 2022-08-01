# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/cameras/camera_pose.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


'''
camera pose
'''

import numpy as np

from geometry import transformations
from geometry.basics import Translation, Rotation, UnstableRotation


class CameraPose():
    def __init__(self, t: Translation, r: Rotation):
        '''
        Translation and rotation are world to camera
        '''
        assert isinstance(t, Translation)
        assert isinstance(r, Rotation) or isinstance(r, UnstableRotation)
        self.t = t
        self.r = r

    def __str__(self):
        string = f'translation: {self.t}, rotation: {self.r}'
        return string

    @classmethod
    def from_world_to_camera(cls, world_to_camera, unstable=False):
        assert isinstance(world_to_camera, np.ndarray)
        assert world_to_camera.shape == (4, 4)
        vec = transformations.translation_from_matrix(world_to_camera).astype(np.float32)
        t = Translation(vec)
        if unstable:
            r = UnstableRotation(world_to_camera)
        else:
            quat = transformations.quaternion_from_matrix(world_to_camera).astype(np.float32)
            r = Rotation(quat)
        return cls(t, r)

    @classmethod
    def from_camera_to_world(cls, camera_to_world, unstable=False):
        assert isinstance(camera_to_world, np.ndarray)
        assert camera_to_world.shape == (4, 4)
        world_to_camera = np.linalg.inv(camera_to_world)
        world_to_camera /= world_to_camera[3, 3]
        return cls.from_world_to_camera(world_to_camera, unstable)

    @property
    def translation_vector(self):
        return self.t.translation_vector

    @property
    def translation_matrix(self):
        return self.t.translation_matrix

    @property
    def quaternion(self):
        '''
        quaternion format (w, x, y, z)
        '''
        return self.r.quaternion

    @property
    def rotation_matrix(self):
        return self.r.rotation_matrix

    @property
    def world_to_camera(self):
        M = np.matmul(self.translation_matrix, self.rotation_matrix)
        M /= M[3, 3]
        return M

    @property
    def world_to_camera_3x4(self):
        return self.world_to_camera[0:3, 0:4]

    @property
    def extrinsic_matrix(self):
        return self.world_to_camera_3x4

    @property
    def camera_to_world(self):
        M = np.linalg.inv(self.world_to_camera)
        M /= M[3, 3]
        return M

    @property
    def camera_to_world_3x4(self):
        return self.camera_to_world[0:3, 0:4]

    @property
    def camera_center_in_world(self):
        return self.camera_to_world[:3, 3]

    @camera_center_in_world.setter
    def camera_center_in_world(self, value):
        c2w = self.camera_to_world
        c2w[:3, 3] = value
        temp_pose = CameraPose.from_camera_to_world(c2w, unstable=True)
        self.t = temp_pose.t

    @property
    def forward(self):
        return self.camera_to_world[:3, 2]

    @property
    def up(self):
        return -self.camera_to_world[:3, 1]

    @property
    def right(self):
        return self.camera_to_world[:3, 0]
