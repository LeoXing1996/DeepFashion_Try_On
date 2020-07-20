import torch
from torch.autograd import Variable
from util.SSIM import SSIM
import cv2
import os
import numpy as np

from data.data_loader import CreateDataLoader
from options.train_options import TrainOptions


opt = TrainOptions().parse()

dataroot = opt.dataroot
phase = opt.phase

save_dir = os.path.join('sample', opt.which_ckpt)
if opt.name:
    save_dir = os.path.join(save_dir, opt.name)
img_dir = os.path.join(save_dir, 'img')

# data_loader = CreateDataLoader(opt)

img_list = [img for img in os.listdir(img_dir) if 'combine' not in img]

ssim_accu = 0
step = 0
SSIM_cal = SSIM()
# import pdb; pdb.set_trace()
for img in img_list:
    img_F = os.path.join(img_dir, img)
    img_R = os.path.join(dataroot, phase+'_img', img)
    img_F_np = cv2.imread(img_F)
    img_T_np = cv2.imread(img_R)
    img_1 = torch.from_numpy(np.rollaxis(img_F_np, 2)).float().unsqueeze(0) / 255.0
    img_2 = torch.from_numpy(np.rollaxis(img_T_np, 2)).float().unsqueeze(0) / 255.0
    img1, img2 = Variable(img_1, requires_grad=False), Variable(img_2, requires_grad=False)
    ssim = SSIM_cal(img1, img2)
    ssim_accu += ssim
    step += 1
    print('{}/{} SSIM: {}'.format(step, 2032, float(ssim)))

with open(os.path.join(save_dir, 'SSIM.txt'), 'w') as file:
    print('total step: ', step, 'SSIM_accu: ', ssim_accu, 'SSIM_avg:', ssim_accu / step)
    file.write(str(float(ssim_accu / step)))
    file.write('\n')
