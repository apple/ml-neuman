# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/utils/utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import os
import random

import numpy as np
import open3d as o3d
import torch
import pytorch3d
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
from tqdm import tqdm

from models.smpl import SMPL, vertices2joints


def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"


def print_notification(content_list, notification_type='NOTIFICATION'):
    print(f'---------------------- {notification_type} ----------------------')
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def fix_randomness(seed=42):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), f'cannot process data type: {type(torch_img)}'
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width
    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), f'cannot process data type: {type(np_img)}'
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError(f'cannot process this image with shape: {np_img.shape}')


def remove_first_axis(batch):
    for k in batch.keys():
        assert batch[k].shape[0] == 1
        batch[k] = batch[k][0]
    return batch


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_cameras(caps, size=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, cap in enumerate(caps):
        color = next(ax._get_lines.prop_cycler)['color']
        pts = np.array(cap.camera_poly(size))
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=color
        )
        for j in range(1, 5):
            ax.plot([pts[0, 0], pts[j, 0]], [pts[0, 1], pts[j, 1]], [pts[0, 2], pts[j, 2]], c=color)
        ax.plot([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]], [pts[1, 2], pts[2, 2]], c=color)
        ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], [pts[2, 2], pts[3, 2]], c=color)
        ax.plot([pts[3, 0], pts[4, 0]], [pts[3, 1], pts[4, 1]], [pts[3, 2], pts[4, 2]], c=color)
        ax.plot([pts[4, 0], pts[1, 0]], [pts[4, 1], pts[1, 1]], [pts[4, 2], pts[1, 2]], c=color)
    set_axes_equal(ax)
    plt.show()


def visualize_SMPL(verts=None, joints=None, line=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if verts is not None:
        ax.scatter(*verts.T, alpha=0.01)
    if joints is not None:
        for i, j in enumerate(joints):
            cc = np.random.rand(3,)
            ax.text(*j.T, str(i), color=cc)
            ax.scatter(*j.T, c=cc)
        ax.legend()
    if line is not None:
        ax.plot3D(*line.T)
    set_axes_equal(ax)
    plt.show()


def visualize_SMPL_w_cameras(verts_list=None, caps=None, rays=None, samples=None, size=1):
    human_pcds = []
    cmap = matplotlib.cm.get_cmap('Spectral')
    if verts_list:
        for i, verts in enumerate(verts_list):
            temp = o3d.geometry.PointCloud()
            temp.points = o3d.utility.Vector3dVector(verts)
            rgba = cmap(i / max(1, (len(verts_list)-1)))
            color = np.zeros_like(verts[:, :3])
            color[:, 0] = rgba[0]
            color[:, 1] = rgba[1]
            color[:, 2] = rgba[2]
            temp.colors = o3d.utility.Vector3dVector(color)
            human_pcds.append(temp)
    other_pcds = []
    if samples:
        for s in samples:
            temp = o3d.geometry.PointCloud()
            temp.points = o3d.utility.Vector3dVector(s[:, :3])
            if s.shape[1] == 6:
                temp.colors = o3d.utility.Vector3dVector(s[:, 3:] / 255)
            other_pcds.append(temp)
    cam_frames = []
    if caps:
        for i, cap in enumerate(caps):
            pts = np.array(cap.camera_poly(size))
            lns = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts)
            line_set.lines = o3d.utility.Vector2iVector(lns)
            cam_frames.append(line_set)
    o3d.visualization.draw_geometries([*other_pcds, *human_pcds, *cam_frames])


def read_obj(path):
    vert = []
    uvs = []
    faces = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line[:2] == 'v ':
                v = line[2:].split()
                v = [float(i) for i in v]
                vert.append(np.array(v))
            elif line[:3] == 'vt ':
                uv = line[3:].split()
                uv = [float(i) for i in uv]
                uvs.append(np.array(uv))
            elif line[:2] == 'f ':
                f = line[2:].split()
                fv = [int(i.split('/')[0]) for i in f]
                ft = [int(i.split('/')[1]) for i in f]
                faces.append(np.array(fv + ft))

    vert = np.array(vert)
    uvs = np.array(uvs)
    faces = np.array(faces) - 1
    return vert, uvs, faces


def safe_load_weights(model, saved_weights):
    try:
        model.load_state_dict(saved_weights)
    except RuntimeError:
        try:
            weights = saved_weights
            weights = {k.replace('module.', ''): v for k, v in weights.items()}
            model.load_state_dict(weights)
        except RuntimeError:
            try:
                weights = saved_weights
                weights = {'module.' + k: v for k, v in weights.items()}
                model.load_state_dict(weights)
            except RuntimeError:
                try:
                    pretrained_dict = saved_weights
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape))}
                    assert len(pretrained_dict) != 0
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    non_match_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
                    notification = []
                    notification += ['pretrained weights PARTIALLY loaded, following are missing:']
                    notification += [str(non_match_keys)]
                    print_notification(notification, 'WARNING')
                except Exception as e:
                    print(f'pretrained weights loading failed {e}')
                    exit()
    print_notification(['weights are safely loaded'])


def add_border_mask(scene, iterations=10):
    for cap in tqdm(scene.captures, total=len(scene.captures), desc='Generating border masks'):
        if iterations > 0:
            cap.border_mask = ndimage.binary_dilation(cap.binary_mask, iterations=iterations).astype(cap.binary_mask.dtype) - cap.binary_mask
        else:
            cap.border_mask = cap.binary_mask - cap.binary_mask


def smpl_verts_to_center_and_up(verts):
    device = torch.device('cpu')
    body_model = SMPL(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/smplx/smpl'),
        gender='neutral',
        device=device
    )
    joints = vertices2joints(body_model.J_regressor, torch.from_numpy(verts)[None]).numpy()[0]
    spine_ind = [0, 3, 6, 9]
    spine = joints[spine_ind]
    center = spine.mean(axis=0)
    _, _, vv = np.linalg.svd(spine - center)
    linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
    linepts += center
    spine_dir = spine[3] - spine[0]
    up = linepts[1] - linepts[0]
    if np.dot(spine_dir, up) < 0:
        up = linepts[0] - linepts[1]
    up /= np.linalg.norm(up)
    return center, up


def add_pytorch3d_cache(scene, device):
    for i, cap in enumerate(scene.captures):
        cap.posed_mesh = pytorch3d.structures.Meshes(
            verts=torch.from_numpy(scene.verts[i])[None],
            faces=torch.from_numpy(scene.faces[:, :3])[None]
        ).to(device)
        cap.can_mesh = pytorch3d.structures.Meshes(
            verts=torch.from_numpy(scene.static_vert[i])[None],
            faces=torch.from_numpy(scene.faces[:, :3])[None]
        ).to(device)
        cap.posed_mesh_cpu = pytorch3d.structures.Meshes(
            verts=torch.from_numpy(scene.verts[i])[None],
            faces=torch.from_numpy(scene.faces[:, :3])[None]
        ).to('cpu')
        cap.can_mesh_cpu = pytorch3d.structures.Meshes(
            verts=torch.from_numpy(scene.static_vert[i])[None],
            faces=torch.from_numpy(scene.faces[:, :3])[None]
        ).to('cpu')


def move_smpls_to_torch(scene, device):
    sources = [scene.verts, scene.Ts]
    for i in range(2):
        source = sources[i]
        temp = []
        temp_cpu = []
        for item in source:
            temp.append(torch.from_numpy(item).float().to(device))
            temp_cpu.append(torch.from_numpy(item).float().to('cpu'))
        if i == 0:
            scene.verts = temp
            scene.verts_cpu = temp_cpu
        if i == 1:
            scene.Ts = temp
            scene.Ts_cpu = temp_cpu
