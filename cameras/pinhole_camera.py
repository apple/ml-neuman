# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/cameras/pinhole_camera.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


'''
pinhole camera
'''


import numpy as np


class PinholeCamera():
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = int(width)
        self.height = int(height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def __str__(self):
        string = f'width: {self.width}, height: {self.height}, fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}'
        return string

    @classmethod
    def from_intrinsic(cls, width, height, mat):
        fx = mat[0, 0]
        fy = mat[1, 1]
        cx = mat[0, 2]
        cy = mat[1, 2]
        return cls(width, height, fx, fy, cx, cy)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def size(self):
        return self.shape

    @property
    def intrinsic_matrix(self):
        mat = np.array([[self.fx,  0.0,     self.cx],
                        [0.0,      self.fy, self.cy],
                        [0.0,      0.0,     1.0]])
        return mat


def resize_pinhole_camera(pinhole_cam, tgt_size):
    _h, _w = tgt_size
    scale_h = _h / pinhole_cam.shape[0]
    scale_w = _w / pinhole_cam.shape[1]
    _cx, _cy = pinhole_cam.cx * scale_w, pinhole_cam.cy * scale_h
    _fx, _fy = pinhole_cam.fx * scale_w, pinhole_cam.fy * scale_h
    cropped_pinhole_cam = PinholeCamera(_w, _h, _fx, _fy, _cx, _cy)
    return cropped_pinhole_cam
