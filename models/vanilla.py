# Code based on nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
# License from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/LICENSE


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(
        self,
        input_dims,
        max_freq,
        N_freqs,
        log_sampling=True,
        include_input=True,
        min_freq=0,
        mapping='posenc'
    ):
        super().__init__()
        self.input_dims = input_dims
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.N_freqs = N_freqs
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.mapping = mapping
        if mapping == 'rotate':
            self.create_rotated_embedding()
        elif mapping == 'posenc':
            self.create_embedding_fn()
        else:
            assert ValueError(mapping)

    def create_rotated_embedding(self):
        out_dim = self.N_freqs * 2 * 3
        bvals = 2.**np.linspace(self.min_freq, self.max_freq, num=self.N_freqs)
        bvals = np.reshape(np.eye(3)*bvals[:, None, None], [len(bvals)*3, 3])
        rot = np.array([[(2**.5)/2, -(2**.5)/2, 0], [(2**.5)/2, (2**.5)/2, 0], [0, 0, 1]])
        bvals = bvals @ rot.T
        rot = np.array([[1, 0, 0], [0, (2**.5)/2, -(2**.5)/2], [0, (2**.5)/2, (2**.5)/2]])
        bvals = bvals @ rot.T
        try:
            self.bvals = torch.from_numpy(bvals).float().cuda()
        except:
            self.bvals = torch.from_numpy(bvals).float()
        if self.include_input:
            out_dim += 3
        self.out_dim = out_dim

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(self.min_freq, self.max_freq, steps=self.N_freqs)
        else:
            assert 0
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    # @torch.no_grad()
    def forward(self, inputs, cur_iter=None):
        if self.mapping == 'rotate':
            assert inputs.shape[-1] == 3
            pts_flat = torch.cat([torch.sin(inputs @ self.bvals.T),
                                  torch.cos(inputs @ self.bvals.T)], axis=-1)
            if self.include_input:
                pts_flat = torch.cat([inputs, pts_flat], -1)
            return pts_flat
        else:
            assert cur_iter is None
            return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(self, depth=8, width=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, scale=1.0, scale_type='no'):
        """ 
        """
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.scale = scale
        self.scale_type = scale_type

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + input_ch, width) for i in range(depth-1)])

        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + width, width//2)])
            self.feature_linear = nn.Linear(width, width)
            self.alpha_linear = nn.Linear(width, 1)
            self.rgb_linear = nn.Linear(width//2, 3)
        else:
            self.output_linear = nn.Linear(width, output_ch)

    def forward(self, input_pts, input_views=None):
        assert input_pts.shape[-1] == self.input_ch
        if not self.use_viewdirs:
            input_views = None
        if input_views is not None:
            assert input_views.shape[-1] == self.input_ch_views
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            assert input_views is not None
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        if self.scale_type == 'no':
            return outputs
        elif self.scale_type == 'linear':
            return outputs * self.scale
        elif self.scale_type == 'tanh':
            return torch.tanh(outputs) * self.scale


class Joiner(nn.Module):
    def __init__(self, pos_pe, dir_pe, nerf):
        super().__init__()
        self.pos_pe = pos_pe
        self.dir_pe = dir_pe
        self.nerf = nerf

    def forward(self, input_pts, input_views=None):
        input_pts = self.pos_pe(input_pts)
        if input_views is not None:
            input_views = self.dir_pe(input_views)
        return self.nerf(input_pts, input_views)


class OffsetNet(nn.Module):
    def __init__(self, pos_pe, nerf):
        super().__init__()
        self.pos_pe = pos_pe
        self.nerf = nerf

    def forward(self, input_pts, cur_iter=None):
        input_pts = self.pos_pe(input_pts, cur_iter)
        return self.nerf(input_pts)


def build_offset_net(opt):
    st_pe = Embedder(
        opt.raw_pos_dim + 1,
        opt.pos_max_freq,
        opt.pos_N_freqs,
        opt.log_sampling,
        opt.include_input,
        min_freq=opt.pos_min_freq,
    )

    offset_net = NeRF(
        depth=opt.nerf_depth,
        width=opt.nerf_width,
        input_ch=st_pe.out_dim,
        input_ch_views=0,
        output_ch=3,
        use_viewdirs=False,
        scale=opt.offset_scale,
        scale_type=opt.offset_scale_type
    )

    offset_net = OffsetNet(st_pe, offset_net)

    if opt.use_cuda:
        offset_net = offset_net.cuda()
    return offset_net


def build_nerf(opt):
    pos_pe = Embedder(
        opt.raw_pos_dim,
        opt.pos_max_freq,
        opt.pos_N_freqs,
        opt.log_sampling,
        opt.include_input,
        min_freq=opt.pos_min_freq,
        mapping=opt.posenc if hasattr(opt, 'posenc') else 'posenc'
    )

    dir_pe = Embedder(
        opt.raw_dir_dim,
        opt.dir_max_freq,
        opt.dir_N_freqs,
        opt.log_sampling,
        opt.include_input,
        mapping=opt.posenc if hasattr(opt, 'posenc') else 'posenc'
    )

    coarse_net = NeRF(
        depth=opt.nerf_depth,
        width=opt.nerf_width,
        input_ch=pos_pe.out_dim,
        input_ch_views=dir_pe.out_dim,
        use_viewdirs=opt.use_viewdirs
    )

    fine_net = NeRF(
        depth=opt.nerf_depth,
        width=opt.nerf_width,
        input_ch=pos_pe.out_dim,
        input_ch_views=dir_pe.out_dim,
        use_viewdirs=opt.use_viewdirs
    )

    coarse_net = Joiner(pos_pe, dir_pe, coarse_net)
    fine_net = Joiner(pos_pe, dir_pe, fine_net)

    if opt.use_cuda:
        coarse_net = coarse_net.cuda()
        fine_net = fine_net.cuda()
    return coarse_net, fine_net
