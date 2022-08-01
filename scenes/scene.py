# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/sfm_scenes/sfm_scenes.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


'''
scene reconstructed from SfM, mainly VSFM and colmap
'''
import os

from tqdm import tqdm


class BaseScene():
    def __init__(self, captures, point_cloud=None):
        self.captures = captures
        self.point_cloud = point_cloud

    def __str__(self):
        string = f'this scene contains {len(self.captures)} captures'
        if self.point_cloud is not None:
            string += f', with {self.point_cloud.shape[0]} points'
        return string

    def read_data_to_ram(self, data_list=['image']):
        print('warning: you are going to use a lot of RAM.')
        sum_bytes = 0.0
        pbar = tqdm(self.captures, desc=f'reading data, memory usage {sum_bytes / (1024.0 * 1024.0):.2f} MB')
        for cap in pbar:
            if 'image' in data_list:
                sum_bytes += cap.read_image_to_ram()
            if 'depth' in data_list:
                sum_bytes += cap.read_depth_to_ram()
            pbar.set_description(f'reading data, memory usage {sum_bytes / (1024.0 * 1024.0):.2f} MB')
        print(f'----- total memory usage: {sum_bytes / (1024.0 * 1024.0):.2f} MB -----')


class ImageFileScene(BaseScene):
    def __init__(self, captures, point_cloud=None):
        super().__init__(captures, point_cloud)
        self.image_path_to_index = {}
        self.fname_to_index_dict = {}
        self._build_img_X_to_index_dict()

    def __getitem__(self, x):
        if isinstance(x, str):
            try:
                return self.captures[self.img_path_to_index_dict[x]]
            except:
                return self.captures[self.fname_to_index_dict[x]]
        else:
            return self.captures[x]

    def _build_img_X_to_index_dict(self):
        assert (self.captures is not None) and (len(self.captures) > 0), 'there is no captures'
        for i, cap in enumerate(self.captures):
            assert cap.image_path not in self.image_path_to_index, 'image already exists'
            self.image_path_to_index[cap.image_path] = i
            assert os.path.basename(cap.image_path) not in self.fname_to_index_dict, 'Image already exists'
            self.fname_to_index_dict[os.path.basename(cap.image_path)] = i


class RigCameraScene(ImageFileScene):
    def __init__(self, captures, num_views, num_cams, point_cloud=None):
        super().__init__(captures, point_cloud)
        self.num_views = num_views
        self.num_cams = num_cams
        self.view_id_to_index = {}
        self.cam_id_to_index = {}
        self._build_id_to_index_dict()

    def __str__(self):
        string = f'this scene is captured by a {self.num_cams} cameras rig, has {self.num_views} views, and in total {len(self.captures)} captures'
        if self.point_cloud is not None:
            string += f', with {self.point_cloud.shape[0]} points'
        return string

    def get_captures_by_view_id(self, view_id):
        assert view_id < self.num_views
        cap_index = self.view_id_to_index[view_id]
        caps = []
        for i in cap_index:
            caps.append(self.captures[i])
        return caps

    def get_captures_by_cam_id(self, cam_id):
        assert cam_id < self.num_cams
        cap_index = self.cam_id_to_index[cam_id]
        caps = []
        for i in cap_index:
            caps.append(self.captures[i])
        return caps

    def get_capture_by_view_cam_id(self, view_id, cam_id):
        assert view_id < self.num_views
        assert cam_id < self.num_cams
        cap_index_0 = self.cam_id_to_index[cam_id]
        cap_index_1 = self.view_id_to_index[view_id]
        cap_index = list(set(cap_index_0) & set(cap_index_1))
        assert len(cap_index) == 1
        return self.captures[cap_index[0]]

    def _build_img_X_to_index_dict(self):
        assert (self.captures is not None) and (len(self.captures) > 0), 'there is no captures'
        for i, cap in enumerate(self.captures):
            assert cap.image_path not in self.image_path_to_index, 'image already exists'
            self.image_path_to_index[cap.image_path] = i
            assert os.path.basename(cap.image_path) not in self.fname_to_index_dict, 'Image already exists'
            self.fname_to_index_dict[os.path.basename(cap.image_path)] = i

    def _build_id_to_index_dict(self):
        assert (self.captures is not None) and (len(self.captures) > 0), 'there is no captures'
        for i, cap in enumerate(self.captures):
            assert cap.view_id < self.num_views
            assert cap.cam_id < self.num_cams
            if cap.view_id not in self.view_id_to_index:
                self.view_id_to_index[cap.view_id] = [i]
            else:
                self.view_id_to_index[cap.view_id].append(i)
            if cap.cam_id not in self.cam_id_to_index:
                self.cam_id_to_index[cap.cam_id] = [i]
            else:
                self.cam_id_to_index[cap.cam_id].append(i)
