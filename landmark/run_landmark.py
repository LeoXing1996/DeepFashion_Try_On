import os
import os.path as op
import argparse
from PIL import Image, ImageDraw
from dataset import LandMarkDataset

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.models import build_landmark_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Fashion Landmark Detector')
    parser.add_argument(
        '--dataroot',
        type=str,
        help='input image path',
        default='../Data_preprocessing')
    parser.add_argument(
        '--datamode',
        default='train',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--tarpath',
        default='../landmark_res',
        help='folder to save images')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='whether debug')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='config.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/space1/leoxing/data/mmFashionCKPT/landmark/vgg-16/latest.pth',
        help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args


def save_img(name, pred_vis, pred_lms, src_path, tar_pth='../landmark_res', r=2):
    img = Image.open(op.join(src_path, name))
    draw = ImageDraw.Draw(img)
    # for i, lm in enumerate(vis_lms):
    #     x = lm[0]
    #     y = lm[1]
    #     draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 0, 0, 0))
    for vis, lm in zip(pred_vis, pred_lms):
        color = (255, 0, 0, 0) if vis > 0.5 else (0, 255, 0, 0)
        x, y = lm[0], lm[1]
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=color)
    img.save(op.join(tar_pth, name))


def cont_coor(pred_lms, img_w, img_h):
    lms = []
    for i in range(pred_lms.shape[0]):
        x, y = pred_lms[i][0], pred_lms[i][1]
        x = float(x * img_w / 224.)
        y = float(y * img_h / 244.)
        lms.append([x, y])
    return lms


def main():
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not op.exists(args.tarpath):
        os.makedirs(args.tarpath)

    dataset = LandMarkDataset(args.dataroot, args.datamode, args.debug)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    total_imgs = len(dataloader)

    # build model and load checkpoint
    model = build_landmark_detector(cfg.model)
    print('model built')
    load_checkpoint(model, args.checkpoint)
    print('load checkpoint from: {}'.format(args.checkpoint))

    model.cuda()
    model.eval()
    # import pdb
    # pdb.set_trace()
    img_src_path = op.join(args.dataroot, '{}_img').format(args.datamode)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img_tensor, img_w, img_h, img_name = data
            img_tensor = img_tensor.cuda()
            pred_vis, pred_lms = model(img_tensor, return_loss=False)
            pred_lms = pred_lms.data.cpu().numpy()
            print(pred_lms)
            pred_lms_cont = cont_coor(pred_lms, img_w[0], img_h[0])
            save_img(img_name[0], pred_vis, pred_lms_cont,
                     src_path=img_src_path, tar_pth=args.tarpath)
            if i % 10 == 0:
                print('{:5d}/{:5d}'.format(i, total_imgs))


if __name__ == '__main__':
    main()
