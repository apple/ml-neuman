#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import os
import time
from utils.render_utils import render_vanilla

import tqdm
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from utils import utils, render_utils, ray_utils
from trainers import base_trainer, tensorboard_helper
from models.vanilla import weight_reset


class NeRFTrainer(base_trainer.BaseTrainer):
    def __init__(
        self,
        opt,
        coarse_net,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        fine_net=None,
        penalize_empty_space=0
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.penalize_empty_space = penalize_empty_space
        if opt.empty_space_loss_fn == 'l1':
            self.empty_space_loss_fn = F.l1_loss
        elif opt.empty_space_loss_fn == 'mse':
            self.empty_space_loss_fn = F.mse_loss

        super().__init__(opt, coarse_net, optimizer, criterion,
                         train_loader, val_loader, fine_net=fine_net)

    def loss_func(self, batch, device):
        batch = utils.remove_first_axis(batch)
        if self.opt.ablate_nerft:
            coarse_time = batch['viewf_list'].repeat(1, self.opt.samples_per_ray)[..., None]
            fine_time = batch['viewf_list'].repeat(1, self.opt.samples_per_ray + self.opt.importance_samples_per_ray)[..., None]
        else:
            coarse_time, fine_time = None, None
        samples = ray_utils.ray_to_samples(
            batch,
            self.opt.samples_per_ray,
            perturb=self.opt.perturb,
            append_t=coarse_time
        )
        _b, _n, _ = samples[0].shape
        pts = samples[0].to(device)
        dirs = samples[1].to(device)
        z_vals = samples[2].to(device)

        out = self.coarse_net(pts, dirs)
        rgb_map, _, _, weights, _ = render_utils.raw2outputs(out, z_vals, dirs[:, 0, :], raw_noise_std=self.opt.raw_noise_std, white_bkg=self.opt.white_bkg)
        coarse_rgb_loss = F.mse_loss(rgb_map, batch['color'].to(device))
        coarse_empty_space_loss = torch.zeros_like(coarse_rgb_loss)
        if self.penalize_empty_space > 0:
            depth = batch['depth'][:, None].repeat(1, _n).to(device)
            closer_mask = z_vals < (depth * self.opt.margin)
            coarse_empty_space_loss += self.empty_space_loss_fn(
                torch.tanh(torch.relu(out[closer_mask][:, 3])),
                torch.zeros_like(out[closer_mask][:, 3])
            ) * self.penalize_empty_space
        if self.fine_net is not None:
            F_pts, F_dirs, F_z_vals = ray_utils.ray_to_importance_samples(batch, z_vals, weights, self.opt.importance_samples_per_ray, device=device, append_t=fine_time)
            F_out = self.fine_net(
                F_pts,
                F_dirs
            )
            F_rgb_map, _, _, F_weights, _ = render_utils.raw2outputs(F_out, F_z_vals, F_dirs[:, 0, :], raw_noise_std=self.opt.raw_noise_std, white_bkg=self.opt.white_bkg)
            fine_rgb_loss = F.mse_loss(F_rgb_map, batch['color'].to(device))
            fine_empty_space_loss = torch.zeros_like(fine_rgb_loss)
            if self.penalize_empty_space > 0:
                F_depth = batch['depth'][:, None].repeat(1, F_z_vals.shape[1]).to(device)
                F_closer_mask = F_z_vals < (F_depth * self.opt.margin)
                fine_empty_space_loss += self.empty_space_loss_fn(
                    torch.tanh(torch.relu(F_out[F_closer_mask][:, 3])),
                    torch.zeros_like(F_out[F_closer_mask][:, 3])
                ) * self.penalize_empty_space
        if out[..., 3].max() <= 0.0 or F_out[..., 3].max() <= 0.0:
            print('bad weights, reinitializing')
            self.coarse_net.apply(weight_reset)
            if self.fine_net is not None:
                self.fine_net.apply(weight_reset)
            coarse_rgb_loss, coarse_empty_space_loss, fine_rgb_loss, fine_empty_space_loss = [torch.tensor(0.0, requires_grad=True).float().to(device)] * 4
        return coarse_rgb_loss, coarse_empty_space_loss, fine_rgb_loss, fine_empty_space_loss

    def validate_batch(self, batch):
        self.optim.zero_grad()
        device = next(self.coarse_net.parameters()).device
        assert self.coarse_net.training is False
        if self.fine_net is not None:
            assert self.fine_net.training is False
        with torch.no_grad():
            coarse_rgb_loss, coarse_empty_space_loss, fine_rgb_loss, fine_empty_space_loss = self.loss_func(batch, device)
            rgb_loss = coarse_rgb_loss + fine_rgb_loss
            empty_space_loss = coarse_empty_space_loss + fine_empty_space_loss
            total_loss = rgb_loss + empty_space_loss
        return {
            'coarse_rgb_loss':         coarse_rgb_loss.data.item(),
            'coarse_empty_space_loss': coarse_empty_space_loss.data.item(),
            'fine_rgb_loss':           fine_rgb_loss.data.item(),
            'fine_empty_space_loss':   fine_empty_space_loss.data.item(),
            'rgb_loss':                rgb_loss.data.item(),
            'empty_space_loss':        empty_space_loss.data.item(),
            'total_loss':              total_loss.data.item(),
        }

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.coarse_net.training
        self.coarse_net.eval()
        if self.fine_net is not None:
            self.fine_net.eval()
        all_loss = {
            'coarse_rgb_loss':         [],
            'coarse_empty_space_loss': [],
            'fine_rgb_loss':           [],
            'fine_empty_space_loss':   [],
            'rgb_loss':                [],
            'empty_space_loss':        [],
            'total_loss':              [],
        }
        for batch_idx, batch in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            cur_loss = self.validate_batch(batch)
            for k, v in cur_loss.items():
                all_loss[k].append(v)
        dummy_cap = self.val_dataset.scene[self.val_dataset.inclusions[len(self.val_dataset.inclusions) // 2]]
        coarse_render, coarse_depth = render_utils.render_vanilla(
            self.coarse_net,
            dummy_cap,
            fine_net=None,
            rays_per_batch=self.opt.rays_per_batch,
            samples_per_ray=self.opt.samples_per_ray,
            white_bkg=self.opt.white_bkg,
            near_far_source='bkg',
            return_depth=True,
            ablate_nerft=self.opt.ablate_nerft
        )
        coarse_depth = np.stack([coarse_depth]*3, -1)
        fine_render, fine_depth = render_utils.render_vanilla(
            self.coarse_net,
            dummy_cap,
            fine_net=self.fine_net,
            rays_per_batch=self.opt.rays_per_batch,
            samples_per_ray=self.opt.samples_per_ray,
            white_bkg=self.opt.white_bkg,
            near_far_source='bkg',
            return_depth=True,
            ablate_nerft=self.opt.ablate_nerft
        )
        fine_depth = np.stack([fine_depth]*3, -1)
        validation_data = {
            'coarse_rgb_loss':         np.array(all_loss['coarse_rgb_loss']).mean(),
            'coarse_empty_space_loss': np.array(all_loss['coarse_empty_space_loss']).mean(),
            'fine_rgb_loss':           np.array(all_loss['fine_rgb_loss']).mean(),
            'fine_empty_space_loss':   np.array(all_loss['fine_empty_space_loss']).mean(),
            'rgb_loss':                np.array(all_loss['rgb_loss']).mean(),
            'empty_space_loss':        np.array(all_loss['empty_space_loss']).mean(),
            'total_loss':              np.array(all_loss['total_loss']).mean(),
            'render':                  utils.np_img_to_torch_img(np.stack([coarse_render, coarse_depth, fine_render, fine_depth])),
        }
        self.push_validation_data(batch, validation_data)
        self.save_model()
        if training:
            self.coarse_net.train()
            if self.fine_net is not None:
                self.fine_net.train()

    def save_model(self):
        save_dict = {
            'epoch':                   self.epoch,
            'iteration':               self.iteration,
            'optim_state_dict':        self.optim.state_dict(),
            'coarse_model_state_dict': self.coarse_net.state_dict(),
        }
        if self.fine_net is not None:
            save_dict['fine_model_state_dict'] = self.fine_net.state_dict()
        torch.save(save_dict, os.path.join(self.out, 'checkpoint.pth.tar'))

    def push_validation_data(self, batch, validation_data):
        render = vutils.make_grid(validation_data['render'], nrow=2, normalize=True, scale_each=True, )
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        for key in validation_data.keys():
            if 'loss' in key:
                tb_datapack.add_scalar({f'val_loss/{key}': validation_data[key]})
        tb_datapack.add_image({'render/val': render})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def train_batch(self, batch):
        '''train for one batch of data
        '''
        self.optim.zero_grad()
        device = next(self.coarse_net.parameters()).device
        coarse_rgb_loss, coarse_empty_space_loss, fine_rgb_loss, fine_empty_space_loss = self.loss_func(batch, device)
        rgb_loss = coarse_rgb_loss + fine_rgb_loss
        empty_space_loss = coarse_empty_space_loss + fine_empty_space_loss
        if self.iteration >= self.opt.delay_iters:
            total_loss = rgb_loss + empty_space_loss
        else:
            total_loss = rgb_loss
        if np.isnan(total_loss.data.item()):
            print('loss is nan during training')
            self.optim.zero_grad()
        else:
            total_loss.backward()
        self.optim.step()
        self.push_training_data(
            batch,
            {
                'coarse_rgb_loss':         coarse_rgb_loss.data.item(),
                'coarse_empty_space_loss': coarse_empty_space_loss.data.item(),
                'fine_rgb_loss':           fine_rgb_loss.data.item(),
                'fine_empty_space_loss':   fine_empty_space_loss.data.item(),
                'rgb_loss':                rgb_loss.data.item(),
                'empty_space_loss':        empty_space_loss.data.item(),
                'total_loss':              total_loss.data.item(),
            },
            self.optim.param_groups[0]['lr']
        )

        ###### update learning rate ######
        if self.opt.lrate_decay is not None:
            decay_rate = 0.1
            decay_steps = self.opt.lrate_decay * 1000
            new_lrate = self.opt.learning_rate * (decay_rate ** (self.iteration / decay_steps))
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lrate
        ###### update empty space penalty ######
        if self.opt.penalize_empty_space > 0:
            self.penalize_empty_space = self.opt.penalize_empty_space * max(0, 1 - (self.iteration / 60000))
        ##################################

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

    def push_training_data(self, batch, losses, lr):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        for key in losses.keys():
            if 'loss' in key:
                tb_datapack.add_scalar({f'train_loss/{key}': losses[key]})
        tb_datapack.add_scalar({'params/lr': lr})
        tb_datapack.add_scalar({'params/penalize_empty_space': self.penalize_empty_space})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.
        '''
        if hasattr(self.opt, 'load_weights'):
            assert self.opt.load_weights is None or self.opt.load_weights == False
        # 1. load check point
        checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError(
                'model check point cannnot found: {0}'.format(checkpoint_path))
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def load_pretrained_weights(self):
        '''
        load pretrained weights from another model
        '''
        assert os.path.isfile(self.opt.load_weights_path), self.opt.load_weights_path
        content_list = []
        saved = torch.load(self.opt.load_weights_path, map_location='cpu')
        utils.safe_load_weights(self.coarse_net, saved['coarse_model_state_dict'])
        content_list += [f'Loaded pretrained coarse_net weights from {self.opt.load_weights_path}']
        if 'fine_model_state_dict' in saved and self.fine_net is not None:
            utils.safe_load_weights(self.fine_net, saved['fine_model_state_dict'])
            content_list += [f'Loaded pretrained fine_net weights from {self.opt.load_weights_path}']
        utils.print_notification(content_list)
