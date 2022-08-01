# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/transformations/transform_basics.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import numpy as np

from geometry import transformations


class Rotation():
    def __init__(self, quat):
        '''
        quaternion format (w, x, y, z)
        '''
        self.quaternion = quat

    def __str__(self):
        string = f'quaternion: {self.quaternion}'
        return string

    @classmethod
    def from_matrix(cls, mat):
        assert isinstance(mat, np.ndarray)
        if mat.shape == (3, 3):
            id_mat = np.eye(4)
            id_mat[0:3, 0:3] = mat
            mat = id_mat
        assert mat.shape == (4, 4)
        quat = transformations.quaternion_from_matrix(mat).astype(np.float32)
        return cls(quat)

    @property
    def rotation_matrix(self):
        return transformations.quaternion_matrix(self.quaternion).astype(np.float32)

    @rotation_matrix.setter
    def rotation_matrix(self, mat):
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (4, 4)
        quat = transformations.quaternion_from_matrix(mat).astype(np.float32)
        self.quaternion = quat

    @property
    def quaternion(self):
        assert isinstance(self._quaternion, np.ndarray)
        assert self._quaternion.shape == (4,)
        assert np.isclose(np.linalg.norm(self._quaternion), 1.0), 'self._quaternion is not normalized or valid'
        return self._quaternion

    @quaternion.setter
    def quaternion(self, quat):
        assert isinstance(quat, np.ndarray)
        assert quat.shape == (4,)
        if not np.isclose(np.linalg.norm(quat), 1.0):
            print(f'WARNING: normalizing the input quatternion to unit quaternion: {np.linalg.norm(quat)}')
            quat = quat / np.linalg.norm(quat)
        assert np.isclose(np.linalg.norm(quat), 1.0), f'input quaternion is not normalized or valid: {quat}'
        self._quaternion = quat


class UnstableRotation():
    def __init__(self, mat):
        '''
        quaternion format (w, x, y, z)
        '''
        assert isinstance(mat, np.ndarray)
        if mat.shape == (3, 3):
            id_mat = np.eye(4)
            id_mat[0:3, 0:3] = mat
            mat = id_mat
        assert mat.shape == (4, 4)
        mat[:3, 3] = 0
        self._rotation_matrix = mat

    def __str__(self):
        string = f'rotation_matrix: {self.rotation_matrix}'
        return string

    @property
    def rotation_matrix(self):
        return self._rotation_matrix


class Translation():
    def __init__(self, vec):
        self.translation_vector = vec

    def __str__(self):
        string = f'translation: {self.translation_vector}'
        return string

    @classmethod
    def from_matrix(cls, mat):
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (4, 4)
        vec = transformations.translation_from_matrix(mat).astype(np.float32)
        return cls(vec)

    @property
    def translation_matrix(self):
        return transformations.translation_matrix(self.translation_vector).astype(np.float32)

    @translation_matrix.setter
    def translation_matrix(self, mat):
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (4, 4)
        vec = transformations.translation_from_matrix(mat).astype(np.float32)
        self.translation_vector = vec

    @property
    def translation_vector(self):
        return self._translation_vector

    @translation_vector.setter
    def translation_vector(self, vec):
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (3,)
        assert vec.dtype == np.float32
        self._translation_vector = vec
