### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import rescale
import numpy as np
import random
import ipdb


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    #flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_PAF_transform():
    transform_list = []
    # here we default to the standard case
    #   opt.resize_or_crop == 'scale_width'
    # PAF is a np array, we want to
    # 1. resize this ndarray and
    # 2. call `ToTensor` --> this function can only provide a `DoubleTensor` output with
    #                        respect to the input data type
    #                        we need to call type(torch.FloatTensor) manually in get_item
    target_width = 192
    transform_list.append(transforms.Lambda(lambda arr: __scale_width_ndarray(arr, target_width)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __scale_width_ndarray(arr, target_width):
    oh, ow = arr.shape[:2]
    if ow == target_width:
        return arr
    target_scale = target_width / ow
    return rescale(arr, target_scale, multichannel=True)


# limbs vector for PAF--> TODO: get only PAFs related to upper points
def kp_connections(keypoints, only_upper=False):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],
        [keypoints.index('neck'), keypoints.index('nose')],
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]
    if not only_upper:  # add pafs with respect of lower bodys
        kp_lines += [
            [keypoints.index('neck'), keypoints.index('right_hip')],
            [keypoints.index('right_hip'), keypoints.index('right_knee')],
            [keypoints.index('right_knee'), keypoints.index('right_ankle')],
            [keypoints.index('neck'), keypoints.index('left_hip')],
            [keypoints.index('left_hip'), keypoints.index('left_knee')],
            [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        ]
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear']

    return keypoints
