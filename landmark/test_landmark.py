from __future__ import division
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.apis import get_root_logger, init_dist, test_landmark_detector
from mmfashion.datasets import get_dataset
from mmfashion.datasets import build_dataloader
from mmfashion.models import build_landmark_detector
from mmcv.parallel import MMDataParallel

import os
import os.path as op
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from dataset import LandmarkDetectDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a Fashion Landmark Detector')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='config.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint/vgg/latest.pth',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training',
        default=True)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


ten_to_pil = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.]),
    transforms.ToPILImage()])


def save_img(img, name, pred_vis, pred_lms, tar_pth='../landmark_res', r=2):
    draw = ImageDraw.Draw(img)
    for vis, lm in zip(pred_vis, pred_lms):
        color = (255, 0, 0, 0) if vis > 0.5 else (0, 255, 0, 0)
        x, y = lm[0], lm[1]
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=color)
    img.save(op.join(tar_pth, name))


def test_detector_vis(model, dataset, cfg):
    data_loader = build_dataloader(
        dataset,
        1,  #  change `cfg.data.imgs_per_gpu` to 1
        cfg.data.workers_per_gpu,
        len(cfg.gpus.test),
        dist=False,
        shuffle=False)
    model = MMDataParallel(model, device_ids=cfg.gpus.test).cuda()
    model.eval()
    # import pdb
    # pdb.set_trace()
    for batch_idx, test_data in enumerate(data_loader):
        if batch_idx == 10:
            break
        img = test_data['img']
        landmark = test_data['landmark_for_regression']
        vis = test_data['vis']

        pred_vis, pred_lms = model(img, return_loss=False)

        img_pil_pred = ten_to_pil(img.squeeze().detach().clone())
        img_pred_name = '{}_pred.png'.format(batch_idx)
        img_pil_gt = ten_to_pil(img.squeeze().detach().clone())
        img_gt_name = '{}_gt.png'.format(batch_idx)
        # save pred
        save_img(img_pil_pred, img_pred_name, pred_vis, pred_lms)
        save_img(img_pil_gt, img_gt_name, vis.view(-1), landmark.view(-1, 2))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed test: {}'.format(distributed))

    # data loader
    # test_dataset = get_dataset(cfg.data.test)
    test_dataset = LandmarkDetectDataset(
        cfg.data.test.img_path, cfg.data.test.img_file, cfg.data.test.bbox_file,
        cfg.data.test.landmark_file, cfg.data.test.img_size
    )
    print('dataset loaded')

    # build model and load checkpoint
    model = build_landmark_detector(cfg.model)
    print('model built')

    load_checkpoint(model, cfg.load_from, map_location='cpu')
    print('load checkpoint from: {}'.format(cfg.load_from))

    # test
    # test_landmark_detector(
    #     model,
    #     test_dataset,
    #     cfg,
    #     distributed=distributed,
    #     validate=args.validate,
    #     logger=logger)

    test_detector_vis(model, test_dataset, cfg)


if __name__ == '__main__':
    main()
