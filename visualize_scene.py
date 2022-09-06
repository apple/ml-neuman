#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Draw SMPL meshes(as vertices) with the scene point cloud

Example:
python visualize_scene.py --scene_dir ./data/citron
'''

import argparse

import imageio
import open3d as o3d
import numpy as np
import matplotlib

from data_io import neuman_helper


def main(opt):
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=None,
        normalize=True,
        smpl_type='optimized'
    )
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene.point_cloud[:, :3])
    scene_pcd.colors = o3d.utility.Vector3dVector(scene.point_cloud[:, 3:] / 255)

    human_pcds = []
    cam_poly_list = []
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i, cap in enumerate(scene.captures):
        temp = o3d.geometry.PointCloud()
        temp.points = o3d.utility.Vector3dVector(scene.verts[i][:, :3])
        rgba = cmap(i / len(scene.captures))
        color = np.zeros_like(scene.verts[i][:, :3])
        color[:, 0] = rgba[0]
        color[:, 1] = rgba[1]
        color[:, 2] = rgba[2]
        temp.colors = o3d.utility.Vector3dVector(color)
        human_pcds.append(temp)
        cam_pts = np.array(cap.camera_poly(0.1))
        lns = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        cam_poly = o3d.geometry.LineSet()
        cam_poly.points = o3d.utility.Vector3dVector(cam_pts)
        cam_poly.lines = o3d.utility.Vector2iVector(lns)
        cam_poly_list.append(cam_poly)
    o3d.visualization.draw_geometries([scene_pcd, *human_pcds, *cam_poly_list])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    opt = parser.parse_args()
    main(opt)
