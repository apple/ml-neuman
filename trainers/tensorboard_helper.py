# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/trainers/tensorboard_helper.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import abc

import tensorboardX


class TensorboardDatapack():
    '''data dictionary for pushing to tb
    '''

    def __init__(self):
        self.SCALAR_NAME = 'scalar'
        self.HISTOGRAM_NAME = 'histogram'
        self.IMAGE_NAME = 'image'
        self.TEXT_NAME = 'text'
        self.datapack = {}
        self.datapack[self.SCALAR_NAME] = {}
        self.datapack[self.HISTOGRAM_NAME] = {}
        self.datapack[self.IMAGE_NAME] = {}
        self.datapack[self.TEXT_NAME] = {}

    def set_training(self, training):
        self.training = training

    def set_iteration(self, iteration):
        self.iteration = iteration

    def add_scalar(self, scalar_dict):
        self.datapack[self.SCALAR_NAME].update(scalar_dict)

    def add_histogram(self, histogram_dict):
        self.datapack[self.HISTOGRAM_NAME].update(histogram_dict)

    def add_image(self, image_dict):
        self.datapack[self.IMAGE_NAME].update(image_dict)

    def add_text(self, text_dict):
        self.datapack[self.TEXT_NAME].update(text_dict)


class TensorboardHelperBase(abc.ABC):
    '''abstract base class for tb helpers
    '''

    def __init__(self, tb_writer):
        self.tb_writer = tb_writer

    @abc.abstractmethod
    def add_data(self, tb_datapack):
        pass


class TensorboardScalarHelper(TensorboardHelperBase):
    def add_data(self, tb_datapack):
        scalar_dict = tb_datapack.datapack[tb_datapack.SCALAR_NAME]
        for key, val in scalar_dict.items():
            self.tb_writer.add_scalar(
                key, val, global_step=tb_datapack.iteration)


class TensorboardHistogramHelper(TensorboardHelperBase):
    def add_data(self, tb_datapack):
        histogram_dict = tb_datapack.datapack[tb_datapack.HISTOGRAM_NAME]
        for key, val in histogram_dict.items():
            self.tb_writer.add_histogram(
                key, val, global_step=tb_datapack.iteration)


class TensorboardImageHelper(TensorboardHelperBase):
    def add_data(self, tb_datapack):
        image_dict = tb_datapack.datapack[tb_datapack.IMAGE_NAME]
        for key, val in image_dict.items():
            self.tb_writer.add_image(
                key, val, global_step=tb_datapack.iteration)


class TensorboardTextHelper(TensorboardHelperBase):
    def add_data(self, tb_datapack):
        text_dict = tb_datapack.datapack[tb_datapack.TEXT_NAME]
        for key, val in text_dict.items():
            self.tb_writer.add_text(
                key, val, global_step=tb_datapack.iteration)


class TensorboardPusher():
    def __init__(self, opt):
        self.tb_writer = tensorboardX.SummaryWriter(opt.tb_dir)
        scalar_helper = TensorboardScalarHelper(self.tb_writer)
        histogram_helper = TensorboardHistogramHelper(self.tb_writer)
        image_helper = TensorboardImageHelper(self.tb_writer)
        text_helper = TensorboardTextHelper(self.tb_writer)
        self.helper_list = [scalar_helper,
                            histogram_helper, image_helper, text_helper]

    def push_to_tensorboard(self, tb_datapack):
        for helper in self.helper_list:
            helper.add_data(tb_datapack)
        self.tb_writer.flush()
