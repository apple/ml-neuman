# Code based on ROMP: https://github.com/Arthur151/ROMP/blob/master/romp/lib/models/smpl.py
# License from ROMP: https://github.com/Arthur151/ROMP/blob/master/LICENSE


# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class SMPL(nn.Module):
    def __init__(self,
                 model_path,
                 gender='neutral',
                 device=None):
        ''' SMPL model constructor'''
        super(SMPL, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        # create the SMPL model
        if osp.isdir(model_path):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_path, model_fn)
        else:
            smpl_path = model_path
        assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
            smpl_path)

        with open(smpl_path, 'rb') as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file,
                                               encoding='latin1'))
        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long, device=device))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype, device=device))

        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(data_struct.shapedirs), dtype=dtype, device=device))

        self.register_buffer('J_regressor',
                             to_tensor(to_np(data_struct.J_regressor), dtype=dtype, device=device))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype, device=device))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0]), device=device).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype, device=device))

    def verts_transformations(self,
                              poses,
                              betas,
                              transl=None,
                              return_tensor=True,
                              concat_joints=False):
        ''' Forward pass for SMPL model but also return transformation of each vertex

        Parameters
        ----------
        betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_tensor: bool, optional
            Return in torch.tensor. (default=True)
        concat_joints: bool, optional
            Concat joints info at the end. (default=False)
        '''
        bn = poses.shape[0]
        assert bn == 1
        if isinstance(poses, np.ndarray):
            poses = to_tensor(poses, self.dtype, self.device)
        if isinstance(betas, np.ndarray):
            betas = to_tensor(betas, self.dtype, self.device)
        if isinstance(transl, np.ndarray):
            transl = to_tensor(transl, self.dtype, self.device)
        L, vertices = lbs(betas, poses, self.v_template,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          return_T=True, concat_joints=concat_joints)
        if transl is not None:
            transl_4x4 = torch.eye(4, dtype=self.dtype, device=self.device)[None]
            transl_4x4[0, :3, 3] = transl.unsqueeze(1)
            T = torch.matmul(transl_4x4, L)
        else:
            T = L
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()[0]
            T = T.detach().cpu().numpy()[0]
        return vertices, T

    def forward(self,
                poses,
                betas,
                transl=None,
                return_joints=False,
                return_tensor=True):
        ''' Forward pass for SMPL model

        Parameters
        ----------
        betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        return_joints: bool, optional
            Return (vertices, joints) tuple. (default=False)
        return_tensor: bool, optional
            Return in torch.tensor. (default=True)
        '''
        bn = poses.shape[0]
        assert bn == 1
        if isinstance(poses, np.ndarray):
            poses = to_tensor(poses, self.dtype, self.device)
        if isinstance(betas, np.ndarray):
            betas = to_tensor(betas, self.dtype, self.device)
        if isinstance(transl, np.ndarray):
            transl = to_tensor(transl, self.dtype, self.device)
        vertices, joints = lbs(betas, poses, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        if transl is not None:
            vertices = vertices + transl.unsqueeze(1)
            joints = joints + transl.unsqueeze(1)
        if not return_tensor:
            vertices = vertices.detach().cpu().numpy()
            joints = joints.detach().cpu().numpy()
        if return_joints:
            return vertices[0], joints[0]
        return vertices[0]


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).contiguous().view(
        batch_size, -1, 3)

    lmk_faces = lmk_faces + torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3).contiguous()[lmk_faces].contiguous().view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32, return_T=False, concat_joints=False):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    if return_T:
        if concat_joints:
            return torch.cat([T, A], dim=1), torch.cat([v_posed, J], dim=1)
        else:
            return T, v_posed

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.contiguous().view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)).contiguous().view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
