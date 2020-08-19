### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        # self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        # self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # self.parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        # self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        # self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        # self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        # self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        # self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")

        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--img_debugger', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--ckpt_base', type=str, default='../ACGPN_train/checkpoints/', help='base dir for checkpoints')
        self.parser.add_argument('--which_ckpt', type=str, default='offical_release', help='which ckpt setting used for evaluation')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--no_img', action='store_true', help='save generate images')

        # for pafs input --> Unet (Warping Network) is not applied to pafs (temp)
        self.parser.add_argument('--pafs_upper', action='store_true', help='only use upper PAFs')
        self.parser.add_argument('--pafs_G1', action='store_true', help='pafs to G1 (semantic mask)')
        self.parser.add_argument('--pafs_G2', action='store_true', help='pafs to G2 (predicted clothes mask)')
        self.parser.add_argument('--pafs_Content', action='store_true', help='pafs to G / content fusion module')

        # options for evaluation
        self.parser.add_argument('--remove_old', action='store_true', help='delete old result text')
        self.parser.add_argument('--eval_inception', action='store_true')
        self.parser.add_argument('--splits', type=int, default=1, help='split for Inception Score')
        self.parser.add_argument('--out_num', type=int, default=1000, help='output number of Inception V3 network')

        self.parser.add_argument('--eval_ssim', action='store_true')
        self.parser.add_argument('--window_size', type=int, default=11)
        self.parser.add_argument('--size_average', type=bool, default=True)
        self.parser.add_argument('--channel', type=int, default=3)

        self.parser.add_argument('--grid_pth', default='../Data_preprocessing/grid.png')
        self.isTrain = True
