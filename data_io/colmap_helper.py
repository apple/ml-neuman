# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/datasets/colmap_helper.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import os
import re
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from geometry.basics import Translation, Rotation
from cameras.camera_pose import CameraPose
from cameras.pinhole_camera import PinholeCamera
from cameras import captures as captures_module
from scenes import scene as scene_module


ImageMeta = namedtuple('ImageMeta', ['image_id', 'camera_pose', 'camera_id', 'image_path'])


class ColmapAsciiReader():
    def __init__(self):
        pass

    @classmethod
    def read_scene(cls, scene_dir, images_dir, tgt_size=None, order='default'):
        point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        captures = cls.read_captures(images_path, cameras_path, images_dir, tgt_size, order)
        point_cloud = cls.read_point_cloud(point_cloud_path)
        scene = scene_module.ImageFileScene(captures, point_cloud)
        return scene

    @staticmethod
    def read_point_cloud(points_txt_path):
        with open(points_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# 3D point list with one line of data per point:\n'
            line = fid.readline()
            assert line == '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
            line = fid.readline()
            assert re.search('^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$', line)
            num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_points = int(num_points)
            mean_track_length = float(mean_track_length)

            xyz = np.zeros((num_points, 3), dtype=np.float32)
            rgb = np.zeros((num_points, 3), dtype=np.float32)

            for i in tqdm(range(num_points), desc='reading point cloud'):
                elems = fid.readline().split()
                xyz[i] = list(map(float, elems[1:4]))
                rgb[i] = list(map(float, elems[4:7]))
            pcd = np.concatenate([xyz, rgb], axis=1)
        return pcd

    @classmethod
    def read_captures(cls, images_txt_path, cameras_txt_path, images_dir, tgt_size, order='default'):
        captures = []
        cameras = cls.read_cameras(cameras_txt_path)
        images_meta = cls.read_images_meta(images_txt_path, images_dir)
        if order == 'default':
            keys = images_meta.keys()
        elif order == 'video':
            keys = []
            frames = []
            for k, v in images_meta.items():
                keys.append(k)
                frames.append(os.path.basename(v.image_path))
            keys = [x for _, x in sorted(zip(frames, keys))]
        else:
            raise ValueError(f'unknown order: {order}')
        for i, key in enumerate(keys):
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = images_meta[key].camera_pose
            cur_image_path = images_meta[key].image_path
            if tgt_size is None:
                cap = captures_module.RGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose)
            else:
                cap = captures_module.ResizedRGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose, tgt_size)
            if order == 'video':
                cap.frame_id = {'frame_id': i, 'total_frames': len(images_meta)}
            captures.append(cap)
        return captures

    @classmethod
    def read_cameras(cls, cameras_txt_path):
        cameras = {}
        with open(cameras_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Camera list with one line of data per camera:\n'
            line = fid.readline()
            assert line == '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
            line = fid.readline()
            assert re.search('^# Number of cameras: \d+\n$', line)
            num_cams = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

            for _ in tqdm(range(num_cams), desc='reading cameras'):
                elems = fid.readline().split()
                camera_id = int(elems[0])
                if elems[1] == 'SIMPLE_RADIAL':
                    width, height, focal_length, cx, cy, radial = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, focal_length, focal_length, cx, cy)
                elif elems[1] == 'PINHOLE':
                    width, height, fx, fy, cx, cy = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, fx, fy, cx, cy)
                elif elems[1] == 'OPENCV':
                    width, height, fx, fy, cx, cy, k1, k2, k3, k4 = list(map(float, elems[2:]))
                    cur_cam = PinholeCamera(width, height, fx, fy, cx, cy)
                else:
                    raise ValueError(f'unsupported camera: {elems[1]}')
                assert camera_id not in cameras
                cameras[camera_id] = cur_cam
        return cameras

    @classmethod
    def read_images_meta(cls, images_txt_path, images_dir):
        images_meta = {}
        with open(images_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Image list with two lines of data per image:\n'
            line = fid.readline()
            assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
            line = fid.readline()
            assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
            line = fid.readline()
            assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
            num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_images = int(num_images)
            mean_ob_per_img = float(mean_ob_per_img)

            for _ in tqdm(range(num_images), desc='reading images meta'):
                elems = fid.readline().split()
                assert len(elems) == 10
                line = fid.readline()
                image_path = os.path.join(images_dir, elems[9])
                assert os.path.isfile(image_path), f'missing file: {image_path}'
                image_id = int(elems[0])
                qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))
                t = Translation(np.array([tx, ty, tz], dtype=np.float32))
                r = Rotation(np.array([qw, qx, qy, qz], dtype=np.float32))
                camera_pose = CameraPose(t, r)
                camera_id = int(elems[8])
                assert image_id not in images_meta, f'duplicated image, id: {image_id}, path: {image_path}'
                images_meta[image_id] = ImageMeta(image_id, camera_pose, camera_id, image_path)
        return images_meta
