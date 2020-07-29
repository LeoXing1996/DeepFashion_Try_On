import os
import os.path as op
from data.base_dataset import BaseDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import cv2
import torch


class SSIMDataset(Dataset):
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.save_dir = op.join('sample', opt.which_ckpt)
        if opt.name:
            self.save_dir = op.join(self.save_dir, opt.name)
        self.img_dir = op.join(self.save_dir, 'img')
        self.img_list = [img for img in os.listdir(self.img_dir) if 'combine' not in img]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def toTensor(self, img):
        img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ten = torch.from_numpy(np.rollaxis(img, 2)).float() / 255.0
        return img_ten

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_F = op.join(self.img_dir, img_name)
        img_R = op.join(self.dataroot, self.phase+'_img', img_name)
        # img_F_ten = self.transform(Image.open(img_F))
        # img_R_ten = self.transform(Image.open(img_R))
        img_F_ten = self.toTensor(img_F)
        img_R_ten = self.toTensor(img_R)
        return {'img_F': img_F_ten, 'img_R': img_R_ten}


class InceptionDataset(Dataset):
    def __init__(self, opt):
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.save_dir = op.join('sample', opt.which_ckpt)
        if opt.name:
            self.save_dir = op.join(self.save_dir, opt.name)
        self.img_dir = op.join(self.save_dir, 'img')
        self.img_list = [img for img in os.listdir(self.img_dir) if 'combine' not in img]
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = op.join(self.img_dir, img_name)
        img = self.transform(Image.open(img))
        return {'img': img}

    def __len__(self):
        return len(self.img_list)
