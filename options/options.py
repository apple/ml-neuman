# Code based on COTR: https://github.com/ubc-vision/COTR/blob/master/COTR/options/options.py
#                       https://github.com/ubc-vision/COTR/blob/master/COTR/options/options_utils.py
# License from COTR: https://github.com/ubc-vision/COTR/blob/master/LICENSE


import sys
import os
import json

from utils import utils


def str2bool(v: str):
    return v.lower() in ('true', '1', 'yes', 'y', 't')


def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    utils.print_notification(content_list, 'OPTIONS')


def opt_to_string(opt):
    string = '\n\n'
    string += 'python ' + ' '.join(sys.argv)
    string += '\n\n'
    args = list(vars(opt))
    args.sort()
    for arg in args:
        string += arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg)) + '\n\n'
    return string


def save_opt(opt):
    '''save options to a json file
    '''
    if not os.path.exists(opt.out):
        os.makedirs(opt.out)
    json_path = os.path.join(opt.out, 'params.json')
    with open(json_path, 'w') as fp:
        json.dump(vars(opt), fp, indent=0, sort_keys=True)


def set_general_option(parser):
    general_opt = parser.add_argument_group('General')
    general_opt.add_argument('--use_cuda', type=str2bool, default=True, help='cuda')


def set_nerf_option(parser):
    nerf_opt = parser.add_argument_group('NeRF')
    nerf_opt.add_argument('--nerf_depth', type=int, default=8, help='network depth')
    nerf_opt.add_argument('--nerf_width', type=int, default=256, help='network width')
    nerf_opt.add_argument('--use_viewdirs', type=str2bool, default=True, help='use directions as network input')
    nerf_opt.add_argument('--specular_can', type=str2bool, default=True, help='no specular in canonical space')


def set_pe_option(parser):
    pe_opt = parser.add_argument_group('Positional Encoding')
    pe_opt.add_argument('--raw_pos_dim', type=int, default=3, help='dimension of postion(XYZ, or XY)')
    pe_opt.add_argument('--pos_min_freq', type=int, default=0, help='')
    pe_opt.add_argument('--pos_max_freq', type=int, default=9, help='')
    pe_opt.add_argument('--pos_N_freqs', type=int, default=10, help='')
    pe_opt.add_argument('--raw_dir_dim', type=int, default=3, help='dimension of direction')
    pe_opt.add_argument('--dir_max_freq', type=int, default=3, help='')
    pe_opt.add_argument('--dir_N_freqs', type=int, default=4, help='')
    pe_opt.add_argument('--log_sampling', type=bool, default=True, help='')
    pe_opt.add_argument('--include_input', type=bool, default=True, help='')
    pe_opt.add_argument('--can_posenc', type=str, default='rotate', help='rotate positional encoding for can space')


def set_render_option(parser):
    render_opt = parser.add_argument_group('Rendering')
    render_opt.add_argument('--rays_per_batch', default=2048, type=int, help='how many rays per batch')
    render_opt.add_argument('--samples_per_ray', default=128, type=int, help='how many samples per ray')
    render_opt.add_argument('--render_h', default=None, type=int, help='image height')
    render_opt.add_argument('--render_w', default=None, type=int, help='image width')
    render_opt.add_argument('--weights_path', required=False, default=None, type=str, help='weights path')
    render_opt.add_argument('--white_bkg', type=str2bool, default=True, required=False)


def set_trajectory_option(parser):
    trajectory_opt = parser.add_argument_group('Trajectory')
    trajectory_opt.add_argument('--trajectory_resolution', default=40, type=int)
