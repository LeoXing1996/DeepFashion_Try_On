### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from util.debug import ImageDebugger
import torch.nn as nn

import cv2
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm

NC = 20


def generate_discrete_label(inputs, label_nc, onehot=True, encode=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    return input_label


# TODO: what does this function do ?
def morpho(mask, iter, bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new = []
    for i in range(len(mask)):
        tem = mask[i].cpu().detach().numpy().squeeze().reshape(256, 192, 1)*255
        tem = tem.astype(np.uint8)
        if bigger:
            tem = cv2.dilate(tem, kernel, iterations=iter)
        else:
            tem = cv2.erode(tem, kernel, iterations=iter)
        tem = tem.astype(np.float64)
        tem = tem.reshape(1, 256, 192)
        new.append(tem.astype(np.float64)/255.0)
    new = np.stack(new)
    new = torch.FloatTensor(new).cuda()
    return new


def encode(label_map, size):
    label_nc = 14
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]
        return loss_filter

    def get_G(self, in_C, out_c, n_blocks, opt, L=1, S=1):
        network = networks.define_G(in_C, out_c, opt.ngf, opt.netG, L, S,
                                    opt.n_downsample_global, n_blocks, opt.n_local_enhancers,
                                    opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        if self.rank:
            return self.apply_dist(network.cuda(self.rank))
        else:
            return network.cuda(self.gpu_ids[0])

    def get_D(self, inc, opt):
        network = networks.define_D(inc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                    opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        return self.to_cuda(network)

    def apply_dist(self, network):
        # add syncBN if necessary
        network = SyncBatchNorm.convert_sync_batchnorm(network)
        network_dist = DDP(network.cuda(self.rank), device_ids=[self.rank])
        # print('Apply dist for on rank : {}'.format(self.rank))
        return network_dist

    def to_cuda(self, network):
        if self.rank is not None:
            return self.apply_dist(network)
        else:
            return network.cuda(self.gpu_ids[0])

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )

        return loss

    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0

            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum()/count
                color[i, 1, :, :] = arms[i, 1, :, :].sum()/count
                color[i, 2, :, :] = arms[i, 2, :, :].sum()/count
        return color

    def initialize(self, opt, rank):
        BaseModel.initialize(self, opt, rank)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.count = 0
        self.perm = torch.randperm(1024*4)

        paf_chns = 26 if opt.pafs_upper else 38
        G1_in_ch, G2_in_ch, G_in_ch = 37, 37, 24
        if opt.pafs_G1:
            G1_in_ch += paf_chns
        if opt.pafs_G2:
            G2_in_ch += paf_chns
        if opt.pafs_Content:
            G_in_ch += paf_chns

        ##### define networks
        self.Unet = self.to_cuda(networks.define_UnetMask(4, self.gpu_ids))
        # G1_in: [pre_clothes_mask, clothes, all_clothes_label, pose, self.gen_noise(shape)]
        # channels: [1,                3,           14,           18,       1] = 37
        self.G1 = self.to_cuda(networks.define_Refine(G1_in_ch, 14, self.gpu_ids))
        # G2_in: [pre_clothes_mask, clothes, masked_label, pose, self.gen_noise(shape)]
        # channels: [1,                3,           14,      18,        1] = 37
        self.G2 = self.to_cuda(networks.define_Refine(G2_in_ch, 1, self.gpu_ids))
        # G_in: [img_hole_hand, masked_label, real_image*clothes_mask, skin_color, self.gen_noise(shape)]
        # channels: [3,               14,               3,                 3,              1]
        self.G = self.to_cuda(networks.define_Refine(G_in_ch, 3, self.gpu_ids))
        #ipdb.set_trace()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        # Discriminator network
        if self.isTrain:
            # D1: input_pool: G1_in                                 --> G1_in_ch
            #     cond_pool: masked_label (real) / arm_label (fake) --> 14
            self.D1 = self.get_D(G1_in_ch+14, opt)
            # D2: input_pool: G2_in                                 --> G2_in_ch
            #     cond_pool: clothes_mask (real) / fake_cl (fake)   --> 1
            self.D2 = self.get_D(G2_in_ch+1, opt)
            # D:  input_pool: G_in                                 --> G_in_ch
            #     cond_pool: real_image (real) / fake_image (fake) --> 3
            self.D = self.get_D(G_in_ch+3, opt)
            # D3: Discriminator for Unet --> not change
            self.D3 = self.get_D(7, opt)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss()
            self.criterionStyle = networks.StyleLoss()
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer ork (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.Unet.parameters())+list(self.G.parameters())+list(self.G1.parameters())+list(self.G2.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=0.0002, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.D3.parameters())+list(self.D.parameters())+list(self.D2.parameters())+list(self.D1.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=0.0002, betas=(opt.beta1, 0.999))

            # load networks
            if not self.isTrain or opt.continue_train or opt.load_pretrain:
                pretrained_path = '' if not self.isTrain else opt.load_pretrain
                self.load_network(self.Unet, 'U', opt.which_epoch, pretrained_path)
                self.load_network(self.G1, 'G1', opt.which_epoch, pretrained_path)
                self.load_network(self.G2, 'G2', opt.which_epoch, pretrained_path)
                self.load_network(self.G, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.D, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.D1, 'D1', opt.which_epoch, pretrained_path)
                self.load_network(self.D2, 'D2', opt.which_epoch, pretrained_path)
                self.load_network(self.D3, 'D3', opt.which_epoch, pretrained_path)
                # seems should load optimizer too ?
                self.load_network(self.optimizer_G, 'OG', opt.which_epoch, pretrained_path)
                self.load_network(self.optimizer_D, 'OD', opt.which_epoch, pretrained_path)

            # optimizer G + B
            #params = list(self.netG.parameters()) + list(self.netB.parameters())
            #self.optimizer_GB = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, clothes_mask, all_clothes_label):
        size = label_map.size()
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        masked_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        masked_label = masked_label.scatter_(1, (label_map*(1-clothes_mask)).data.long().cuda(), 1.0)

        c_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_label = c_label.scatter_(1, all_clothes_label.data.long().cuda(), 1.0)

        input_label = Variable(input_label)

        return input_label, masked_label, c_label

    def encode_input_test(self, label_map, label_map_ref, real_image_ref, infer=False):

        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
            input_label_ref = label_map_ref.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            input_label_ref = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
                input_label_ref = input_label_ref.half()

        input_label = Variable(input_label, volatile=infer)
        input_label_ref = Variable(input_label_ref, volatile=infer)
        real_image_ref = Variable(real_image_ref.data.cuda())

        return input_label, input_label_ref, real_image_ref

    def discriminate(self, netD, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()

    def forward(self, label, pre_clothes_mask,
                img_fore, clothes_mask,
                clothes, all_clothes_label,
                real_image, pose, pafs, mask):
        # Some input:
        # img_fore -> foreground of `real_images`
        # clothes_mask -> target (part of `label` == 4 or cloth in real image)
        # pre_clothes_mask -> source (`edge` or cloth mask without warping)
        # mask -> ??? TODO

        # Encode Inputs --> change mask to one-hot label
        #   input_label: real_image mask with all part
        #   masked_label: mask w/o cloth part (M_W^S)
        #   all_clothes_label: combine `arm` and `cloth` as `cloth` label, fused map in paper (M^F)
        input_label, masked_label, all_clothes_label = self.encode_input(label, clothes_mask, all_clothes_label)

        arm1_mask = torch.FloatTensor((label.cpu().numpy() == 11).astype(np.float)).cuda()
        arm2_mask = torch.FloatTensor((label.cpu().numpy() == 13).astype(np.float)).cuda()
        pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = clothes*pre_clothes_mask

        # fake_image,warped,warped_mask=self.Unet(clothes,clothes_mask,pre_clothes_mask)
        # real_image=real_image * clothes_mask+(1-clothes_mask)*-1
        shape = pre_clothes_mask.shape

        if self.opt.pafs_G1:
            G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label,
                               pose, pafs, self.gen_noise(shape)], dim=1)
        else:
            G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label,
                               pose, self.gen_noise(shape)], dim=1)
        arm_label = self.G1(G1_in)
        arm_label = self.sigmoid(arm_label)
        CE_loss = self.cross_entropy2d(arm_label, (label * (1 - clothes_mask)).transpose(0, 1)[0].long())*10

        armlabel_map = generate_discrete_label(arm_label.detach(), 14, False)
        dis_label = generate_discrete_label(arm_label.detach(), 14)

        if self.opt.pafs_G2:
            G2_in = torch.cat([pre_clothes_mask, clothes, masked_label,
                               pose, pafs, self.gen_noise(shape)], dim=1)
        else:
            G2_in = torch.cat([pre_clothes_mask, clothes, masked_label,
                               pose, self.gen_noise(shape)], dim=1)

        fake_cl = self.G2(G2_in)
        fake_cl = self.sigmoid(fake_cl)
        CE_loss += self.BCE(fake_cl, clothes_mask)*10

        #ipdb.set_trace()
        fake_cl_dis = torch.FloatTensor((fake_cl.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        new_arm1_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 11).astype(np.float)).cuda()
        new_arm2_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 13).astype(np.float)).cuda()
        arm1_occ = clothes_mask*new_arm1_mask
        arm2_occ = clothes_mask*new_arm2_mask
        arm1_full = arm1_occ+(1-clothes_mask)*arm1_mask
        arm2_full = arm2_occ+(1-clothes_mask)*arm2_mask
        armlabel_map *= (1-new_arm1_mask)
        armlabel_map *= (1-new_arm2_mask)
        armlabel_map = armlabel_map*(1-arm1_full)+arm1_full*11
        armlabel_map = armlabel_map*(1-arm2_full)+arm2_full*13

        ## construct full label map
        armlabel_map = armlabel_map*(1-fake_cl_dis)+fake_cl_dis*4

        fake_c, warped, warped_mask, rx, ry, cx, cy, rg, cg = \
            self.Unet(clothes, clothes_mask, pre_clothes_mask)
        #ipdb.set_trace()
        # fake_c --> 4 channel
        # tanh(0~2) : T_c^R  (refined cloth in Eq 6)
        # sigmoid(3): \alpha (mask in Eq 6)
        composition_mask = fake_c[:, 3, :, :]
        fake_c = fake_c[:, 0:3, :, :]
        fake_c = self.tanh(fake_c)
        composition_mask = self.sigmoid(composition_mask)

        skin_color = self.ger_average_color((arm1_mask+arm2_mask-arm2_mask*arm1_mask),
                                            (arm1_mask+arm2_mask-arm2_mask*arm1_mask)*real_image)

        # img_hole_hand --> gt of `I_W` in Eq 9
        img_hole_hand = img_fore*(1-clothes_mask)*(1-arm1_mask)*(1-arm2_mask) + \
            img_fore*arm1_mask*(1-mask) + img_fore*arm2_mask*(1-mask)

        if self.opt.e2eContent:
            comp_fake_c = fake_c*(1-composition_mask).unsqueeze(1) + \
                (composition_mask.unsqueeze(1))*warped
            if self.opt.pafs_Content:
                G_in = torch.cat([img_hole_hand, masked_label, comp_fake_c,
                                  skin_color, pafs, self.gen_noise(shape)], dim=1)
            else:
                G_in = torch.cat([img_hole_hand, masked_label, comp_fake_c,
                                  skin_color, self.gen_noise(shape)], dim=1)
            fake_image = self.G(G_in)
            fake_image = self.tanh(fake_image)
        else:
            comp_fake_c = fake_c.detach()*(1-composition_mask).unsqueeze(1) + \
                (composition_mask.unsqueeze(1))*warped.detach()
            if self.opt.pafs_Content:
                G_in = torch.cat([img_hole_hand, masked_label, real_image*clothes_mask,
                                  skin_color, pafs, self.gen_noise(shape)], dim=1)
            else:
                G_in = torch.cat([img_hole_hand, masked_label, real_image*clothes_mask,
                                  skin_color, self.gen_noise(shape)], dim=1)
            fake_image = self.G(G_in.detach())
            fake_image = self.tanh(fake_image)

        ## THE POOL TO SAVE IMAGES\
        input_pool = [G1_in, G2_in, G_in, torch.cat([clothes_mask, clothes], 1)]        ##fake_cl_dis to replace
        #ipdb.set_trace()
        real_pool = [masked_label, clothes_mask, real_image, real_image*clothes_mask]
        fake_pool = [arm_label, fake_cl, fake_image, fake_c]
        D_pool = [self.D1, self.D2, self.D, self.D3]
        pool_lenth = len(fake_pool)
        loss_D_fake = 0
        loss_D_real = 0
        loss_G_GAN = 0
        loss_G_GAN_Feat = 0

        for iter_p in range(pool_lenth):

            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(D_pool[iter_p], input_pool[iter_p].detach(), fake_pool[iter_p], use_pool=True)
            loss_D_fake += self.criterionGAN(pred_fake_pool, False)
            # Real Detection and Loss
            pred_real = self.discriminate(D_pool[iter_p], input_pool[iter_p].detach(), real_pool[iter_p])
            loss_D_real += self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = D_pool[iter_p].forward(torch.cat((input_pool[iter_p].detach(), fake_pool[iter_p]), dim=1))
            loss_G_GAN += self.criterionGAN(pred_fake, True)
            if iter_p < 2:
                continue
            # # GAN feature matching loss
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        #ipdb.set_trace()

        # VGG feature matching loss
        loss_G_VGG = 0
        loss_G_VGG += self.criterionVGG.warp(warped, real_image*clothes_mask) + \
            self.criterionVGG.warp(comp_fake_c, real_image*clothes_mask) * 10
        loss_G_VGG += self.criterionVGG.warp(fake_c, real_image*clothes_mask) * 20
        loss_G_VGG += self.criterionVGG(fake_image, real_image) * 10

        L1_loss = self.criterionFeat(fake_image, real_image)
        # here we modified all L1 loss with weight in `self.opt`
        L1_loss += self.criterionFeat(warped_mask, clothes_mask) * self.opt.warpedMask
        L1_loss += self.criterionFeat(warped, real_image*clothes_mask) * self.opt.warpedCloth
        L1_loss += self.criterionFeat(fake_cl, clothes_mask) * self.opt.predMask  # add L2 loss for predicted_mask
        L1_loss += self.criterionFeat(fake_c, real_image*clothes_mask) * self.opt.unetCloth
        L1_loss += self.criterionFeat(comp_fake_c, real_image*clothes_mask) * self.opt.refinedCloth
        L1_loss += self.criterionFeat(composition_mask, clothes_mask) * self.opt.compMask

        # old loss codes
        # L1_loss += self.criterionFeat(warped_mask, clothes_mask) + \
        #     self.criterionFeat(warped, real_image*clothes_mask)  # L4
        # L1_loss += self.criterionFeat(fake_cl, clothes_mask)  # add L2 loss for predicted_mask
        # L1_loss += self.criterionFeat(fake_c, real_image*clothes_mask)*0.2
        # G_in = torch.cat([img_hole_hand, masked_label, real_image*clothes_mask, skin_color, self.gen_noise(shape)], 1)
        # L1_loss += self.criterionFeat(comp_fake_c, real_image*clothes_mask)*10
        # L1_loss += self.criterionFeat(composition_mask, clothes_mask)

        #
        # style_loss=self.criterionStyle(fake_image, real_image)*200

        # loss_G_GAN_Feat=L1_loss
        style_loss = L1_loss
        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                fake_c, comp_fake_c, dis_label, L1_loss, style_loss, fake_cl, warped, clothes, CE_loss,
                rx*0.1, ry*0.1, cx*0.1, cy*0.1, rg*0.1, cg*0.1]

    def save(self, which_epoch):
        self.save_network(self.Unet, 'U', which_epoch, self.gpu_ids)
        self.save_network(self.G, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.G1, 'G1', which_epoch, self.gpu_ids)
        self.save_network(self.G2, 'G2', which_epoch, self.gpu_ids)
        self.save_network(self.D, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.D1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.D2, 'D2', which_epoch, self.gpu_ids)
        self.save_network(self.D3, 'D3', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_G, 'OG', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_D, 'OD', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def name(self):
        return 'InferenceModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt, None)
        self.isTrain = True  # temply fix isTrain as True
        self.count = 0
        self.debugger = ImageDebugger(opt)

        paf_chns = 26 if opt.pafs_upper else 38
        G1_in_ch, G2_in_ch, G_in_ch = 37, 37, 24
        if opt.pafs_G1:
            G1_in_ch += paf_chns
        if opt.pafs_G2:
            G2_in_ch += paf_chns
        if opt.pafs_Content:
            G_in_ch += paf_chns
        ##### define networks
        with torch.no_grad():
            self.Unet = self.to_cuda(networks.define_UnetMask(4, self.gpu_ids)).eval()
            self.G1 = self.to_cuda(networks.define_Refine(G1_in_ch, 14, self.gpu_ids)).eval()
            self.G2 = self.to_cuda(networks.define_Refine(G2_in_ch, 1, self.gpu_ids)).eval()
            self.G = self.to_cuda(networks.define_Refine(G_in_ch, 3, self.gpu_ids)).eval()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        pretrained_path = os.path.join(opt.ckpt_base, opt.which_ckpt)
        self.load_network(self.Unet, 'U', opt.which_epoch, pretrained_path)
        self.load_network(self.G1, 'G1', opt.which_epoch, pretrained_path)
        self.load_network(self.G2, 'G2', opt.which_epoch, pretrained_path)
        self.load_network(self.G, 'G', opt.which_epoch, pretrained_path)

    def update_debugger(forward_func):
        def update(*args):
            print(args[-2])
            name = args[-2]
            args[0].debugger.set_img_name(name)
            return forward_func(*args)
        return update

    @update_debugger
    def forward(self, label, pre_clothes_mask,
                img_fore, clothes_mask,
                clothes, all_clothes_label,
                real_image, pose,
                grid, mask_fore, pafs, name, debugger):
        input_label, masked_label, all_clothes_label = self.encode_input(label, clothes_mask, all_clothes_label)
        arm1_mask = torch.FloatTensor((label.cpu().numpy() == 11).astype(np.float)).cuda()
        arm2_mask = torch.FloatTensor((label.cpu().numpy() == 13).astype(np.float)).cuda()
        pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = clothes * pre_clothes_mask

        shape = pre_clothes_mask.shape

        if self.opt.pafs_G1:
            G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label,
                               pose, pafs, self.gen_noise(shape)], dim=1)
        else:
            G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label,
                               pose, self.gen_noise(shape)], dim=1)

        arm_label = self.G1(G1_in)

        arm_label = self.sigmoid(arm_label)

        armlabel_map = generate_discrete_label(arm_label.detach(), 14, False)
        dis_label = generate_discrete_label(arm_label.detach(), 14)

        # TODO: here we can compare `armlabel_map` and `label` --> compare result of G1
        # save_lab_tensor(armlabel_map) and save_lab_tensor(label)

        # masked_label is GT used in train stage
        # change `masked_label` to `dis_label` which is generated from G1
        if self.opt.pafs_G2:
            G2_in = torch.cat([pre_clothes_mask, clothes, dis_label,
                               pose, pafs, self.gen_noise(shape)], dim=1)
        else:
            G2_in = torch.cat([pre_clothes_mask, clothes, dis_label,
                               pose, self.gen_noise(shape)], dim=1)

        fake_cl = self.G2(G2_in)
        fake_cl = self.sigmoid(fake_cl)

        fake_cl_dis = torch.FloatTensor((fake_cl.cpu().numpy() > 0.5).astype(np.float)).cuda()
        fake_cl_dis = morpho(fake_cl_dis, 1, True)

        # TODO: here we can compare `fake_cl` and `???` --> compare result of G2 / warping
        # save_rgb_tensor(fake_cl) and save_rgb_tensor
        self.debugger.save_rgb_tensor(fake_cl, 'clothMask_Warp')
        self.debugger.save_rgb_tensor(label == 4, 'clothMask_GT')
        self.debugger.save_diff_tensor(fake_cl, label == 4, 'clothMask_diff')

        new_arm1_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 11).astype(np.float)).cuda()
        new_arm2_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 13).astype(np.float)).cuda()
        fake_cl_dis = fake_cl_dis*(1 - new_arm1_mask)*(1 - new_arm2_mask)
        fake_cl_dis *= mask_fore

        arm1_occ = clothes_mask * new_arm1_mask
        arm2_occ = clothes_mask * new_arm2_mask
        bigger_arm1_occ = morpho(arm1_occ, 10)
        bigger_arm2_occ = morpho(arm2_occ, 10)
        arm1_full = arm1_occ + (1 - clothes_mask) * arm1_mask
        arm2_full = arm2_occ + (1 - clothes_mask) * arm2_mask
        armlabel_map *= (1 - new_arm1_mask)
        armlabel_map *= (1 - new_arm2_mask)
        armlabel_map = armlabel_map * (1 - arm1_full) + arm1_full * 11
        armlabel_map = armlabel_map * (1 - arm2_full) + arm2_full * 13
        armlabel_map *= (1-fake_cl_dis)
        dis_label = encode(armlabel_map, armlabel_map.shape)

        self.debugger.save_rgb_tensor(clothes, 'cloth_1_before_STN')

        fake_c, warped, warped_mask, warped_grid = \
            self.Unet(clothes, fake_cl_dis, pre_clothes_mask, grid)

        self.debugger.save_rgb_tensor(warped, 'cloth_2_after_STN')

        mask = fake_c[:, 3, :, :]
        mask = self.sigmoid(mask)*fake_cl_dis
        fake_c = self.tanh(fake_c[:, 0:3, :, :])

        self.debugger.save_rgb_tensor(fake_c, 'cloth_3_after_refine')

        fake_c = fake_c*(1-mask)+mask*warped

        self.debugger.save_rgb_tensor(fake_c, 'cloth_4_after_comp')
        self.debugger.save_rgb_tensor(real_image*clothes_mask, 'cloth_0_GTl')

        skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                            (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * real_image)
        occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask + clothes_mask)) * \
            (1 - bigger_arm2_occ * (arm2_mask + arm1_mask + clothes_mask))
        img_hole_hand = img_fore * (1 - clothes_mask) * occlude * (1 - fake_cl_dis)
        # change `comp_fake_c` to `fake_c`
        # change `masked_label` to `dis_label`
        if self.opt.pafs_Content:
            G_in = torch.cat([img_hole_hand, dis_label, fake_c,
                              skin_color, pafs, self.gen_noise(shape)], dim=1)
        else:
            G_in = torch.cat([img_hole_hand, dis_label, fake_c,
                              skin_color, self.gen_noise(shape)], dim=1)

        # G_in = torch.cat([img_hole_hand, dis_label, fake_c, skin_color, self.gen_noise(shape)], 1)
        fake_image = self.G(G_in)
        fake_image = self.tanh(fake_image)

        return [fake_image, real_image,
                clothes, real_image*clothes_mask, fake_c,
                arm_label, fake_cl]
