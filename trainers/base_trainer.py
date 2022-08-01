# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/trainers/base_trainer.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import os
import math
import abc
import time

import tqdm
# import tensorboardX

from trainers import tensorboard_helper
from options import options


class BaseTrainer(abc.ABC):
    '''base trainer class.
    contains methods for training, validation, and writing output.
    '''

    def __init__(self, opt, coarse_net, optimizer, criterion,
                 train_loader, val_loader, fine_net=None):
        self.opt = opt
        self.use_cuda = opt.use_cuda
        self.coarse_net = coarse_net
        self.fine_net = fine_net
        self.optim = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = opt.out
        if not os.path.exists(opt.out):
            os.makedirs(opt.out)
        self.epoch = 0
        self.iteration = 0
        self.max_iter = opt.max_iter
        self.valid_iter = opt.valid_iter
        self.tb_pusher = tensorboard_helper.TensorboardPusher(opt)
        self.push_opt_to_tb()
        self.need_resume = opt.resume
        if self.need_resume:
            self.resume()
        if self.opt.load_weights:
            self.load_pretrained_weights()

    def push_opt_to_tb(self):
        opt_str = options.opt_to_string(self.opt)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_text({'options': opt_str})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    @abc.abstractmethod
    def validate_batch(self, data_pack):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
        pass

    def train_epoch(self):
        '''train for one epoch
        one epoch is iterating the whole training dataset once
        '''
        self.coarse_net.train()
        if self.fine_net is not None:
            self.fine_net.train()
        for batch_idx, data_pack in tqdm.tqdm(enumerate(self.train_loader),
                                              initial=self.iteration % len(
                                                  self.train_loader),
                                              total=len(self.train_loader),
                                              desc='Train epoch={0}'.format(
                                                  self.epoch),
                                              ncols=80,
                                              leave=True,
                                              ):

            if self.iteration % self.valid_iter == 0:
                time.sleep(2)  # Prevent possible deadlock during epoch transition
                self.validate()
            self.train_batch(data_pack)

            if self.iteration >= self.max_iter:
                break
            self.iteration += 1

    def train(self):
        '''entrance of the whole training process
        '''
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch,
                                 max_epoch,
                                 desc='Train',
                                 ncols=80):
            self.epoch = epoch
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

    @abc.abstractmethod
    def resume(self):
        pass
