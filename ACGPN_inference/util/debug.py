import numpy as np
from PIL import Image
from matplotlib import cm
import os
import os.path as op
import torch


# TODO
class ImageDebugger:
    def __init__(self, opt, cmap=None):
        save_dir = op.join('sample', opt.name)
        self.base_dir = op.join(save_dir, 'debug')
        CMAP = 'tab20' if cmap is None else cmap
        color_map = cm.get_cmap(CMAP)
        self.colors = color_map.colors


# define color_map for vis
CMAP = 'tab20'
color_map = cm.get_cmap(CMAP)
colors = color_map.colors


def check_and_make_dir(dir):
    if not op.exists(dir):
        os.makedirs(dir)


def get_name_base(base_dir, name, sub_dir):
    base_dir = 'debug_img' if base_dir is None else base_dir
    if sub_dir:
        name_base_dir = op.join(base_dir, sub_dir[0])
        check_and_make_dir(name_base_dir)
        name_base = op.join(name_base_dir, '{}.png')
    else:
        name_base = 'debug_img/{}.png'
    return name_base


# for debug
def save_rgb_tensor(ten, name='debug', base_dir=None, dir=None):
    name_base = get_name_base(base_dir, name, dir)
    ten = ten.cpu() if ten.is_cuda else ten
    ten_np = ten.numpy()[0]
    ten_np = np.rollaxis(ten_np, 0, 3)
    if ten_np.shape[-1] == 1:
        ten_np = np.concatenate((ten_np, ten_np, ten_np), axis=2)
    ten_np = (ten_np + 1) / 2 * 255
    ten_pil = Image.fromarray(ten_np.astype(np.uint8))
    ten_pil.save(name_base.format(name))


def save_lab_tensor(ten, name='debug', base_dir=None, dir=None, cmap='tab20'):
    # this function only used to [1, h, w] label tensor
    name_base = get_name_base(base_dir, name, dir)
    ten = ten.cpu() if ten.is_cuda else ten
    ten_np = np.squeeze(ten.numpy())
    ten_rgb = np.zeros((ten.size(2), ten.size(3), 3))
    for lab in range(14):
        target_color = colors[lab] if lab != 0 else (0, 0, 0)
        ten_rgb[ten_np == lab] = target_color
    ten_pil = Image.fromarray((ten_rgb * 255).astype(np.uint8))
    ten_pil.save(name_base.format(name))


# this function diff of two tensor
def save_diff_tensor(ten1, ten2, name='debug', base_dir=None, dir=None):
    name_base = get_name_base(base_dir, name, dir)
    lab1, lab2 = colors[0], colors[6]
    lab_overlap = colors[5]
    if ten1.dtype != torch.uint8:
        ten1 = ten1 > 0.5
    if ten2.dtype != torch.uint8:
        ten2 = ten2 > 0.5
    ten1_np = np.squeeze(ten1.cpu().numpy())
    ten2_np = np.squeeze(ten2.cpu().numpy())

    overlap_np = ((ten1_np == 1) * (ten2_np == 1))
    ten_canv = np.zeros((ten1_np.shape[0], ten1_np.shape[1], 3))
    ten_canv[ten1_np == 1] = lab1
    ten_canv[ten2_np == 2] = lab2
    ten_canv[overlap_np] = lab_overlap
    ten_pil = Image.fromarray((ten_canv * 255).astype(np.uint8))
    ten_pil.save(name_base.format(name))

