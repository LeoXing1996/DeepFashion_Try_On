### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.save_options import show_opt
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
import datetime
# import ipdb
import torch.distributed as dist

# import pdb
# pdb.set_trace()

SIZE = 320
NC = 14


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def morpho(mask, iter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new = []
    for i in range(len(mask)):
        tem = mask[i].squeeze().reshape(256, 192, 1)*255
        tem = tem.astype(np.uint8)
        tem = cv2.dilate(tem, kernel, iterations=iter)
        tem = tem.astype(np.float64)
        tem = tem.reshape(1, 256, 192)
        new.append(tem.astype(np.float64)/255.0)
    new = np.stack(new)
    return new


def generate_label_color(inputs, label_nc):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], label_nc))
        # label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img*(1-mask)
    M_c = (1-mask.cuda())*M_f
    M_c = M_c+torch.zeros(img.shape).cuda()  ##broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    # check=check>0
    # print(check)
    masked_label = label*(1-mask)
    masked_edge = mask*edge
    masked_color_strokes = mask*(1-color_mask)*color
    masked_noise = mask*noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


# TODO: here old_label is `data['label']`, we can change the input of this function to data only
def changearm(old_label, data):
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


def handle_DDP(rank):
    if rank is not None:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)


def save_model(model, epoch, rank=None):
    if rank is None:
        model.module.save(epoch)
    else:
        model.save(epoch)


def debug_ddp_grad(model, step, rank, show_num=10):
    for name, ten in model.named_parameters():
        print('Step: {} Rank: {} Name: {} Grad: {}'.format(step, rank, name, ten.view(-1)[:show_num]))
        break


def main():
    os.makedirs('sample', exist_ok=True)
    opt = TrainOptions().parse()
    rank = opt.local_rank
    MAIN_DEVICE = rank is None or rank == 0
    handle_DDP(rank)
    torch.backends.cudnn.benchmark = True

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir) and MAIN_DEVICE:
        os.makedirs(expr_dir)
        show_opt(opt, save_dir=expr_dir)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    logger_dir = os.path.join('runs', opt.name)
    img_path = os.path.join('sample', opt.name, 'train')
    img_base = os.path.join('sample', opt.name, 'train', '{}.jpg')
    if not os.path.exists(img_path) and MAIN_DEVICE:
        os.makedirs(img_path)

    writer = SummaryWriter(logger_dir) if MAIN_DEVICE else None

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except FileNotFoundError:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    data_loader = CreateDataLoader(opt, rank)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    if MAIN_DEVICE:
        print('#training images = %d' % dataset_size)

    model = create_model(opt)

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    # display_delta = total_steps % opt.display_freq
    save_delta = total_steps % opt.save_latest_freq

    step = 0
    step_per_batch = dataset_size / opt.batchSize
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            ##add gaussian noise channel && wash the label
            t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
            data['label'] = data['label']*(1-t_mask)+t_mask*4
            mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
            mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
            img_fore = data['image']*mask_fore
            # img_fore_wc = img_fore*mask_fore
            all_clothes_label = changearm(data['label'], data)  # fused map in paper ?
            ############## Forward Pass ######################
            losses, fake_image, real_image, input_label, L1_loss, style_loss, \
                clothes_mask, warped, refined, CE_loss, \
                rx, ry, cx, cy, rg, cg = model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()),
                                                Variable(img_fore.cuda()), Variable(mask_clothes.cuda()),
                                                Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()),
                                                Variable(data['image'].cuda()), Variable(data['pose'].cuda()),
                                                Variable(data['pafs'].cuda()), Variable(data['mask'].cuda()))

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            if rank is None:
                loss_dict = dict(zip(model.module.loss_names, losses))
            else:  # DDP model has no `module` attribute
                loss_dict = dict(zip(model.loss_names, losses))
            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN']+loss_dict.get('G_GAN_Feat', 0) + \
                loss_dict.get('G_VGG', 0)+torch.mean(L1_loss+CE_loss+rx+ry+cx+cy+rg+cg)

            ############### Backward Pass ####################
            if rank is None:
                model.module.optimizer_G.zero_grad()
                loss_G.backward()
                model.module.optimizer_G.step()

                model.module.optimizer_D.zero_grad()
                loss_D.backward()
                model.module.optimizer_D.step()
            else:  # optimize step for DDP
                model.optimizer_G.zero_grad()
                loss_G.backward()
                model.optimizer_G.step()

                model.optimizer_D.zero_grad()
                loss_D.backward()
                model.optimizer_D.step()

                if opt.debug:  # here we try to check grad of model.G
                    debug_ddp_grad(model.G1, step, rank)

            ############## Display results and errors ##########
            if step % opt.display_freq == 0 and MAIN_DEVICE:
                a = generate_label_color(generate_label_plain(input_label), opt.label_nc).float().cuda()
                b = real_image.float().cuda()
                c = fake_image.float().cuda()
                d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
                e = warped
                f = refined
                combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1)/2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img*255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_base.format(str(step)), bgr)

                writer.add_scalar('loss_d', loss_D, step)
                writer.add_scalar('loss_g', loss_G, step)
                writer.add_scalar('loss_L1', torch.mean(L1_loss), step)
                writer.add_scalar('CE_loss', torch.mean(CE_loss), step)
                writer.add_scalar('rx', torch.mean(rx), step)
                writer.add_scalar('ry', torch.mean(ry), step)
                writer.add_scalar('cx', torch.mean(cx), step)
                writer.add_scalar('cy', torch.mean(cy), step)

                writer.add_scalar('loss_g_gan', loss_dict['G_GAN'], step)
                writer.add_scalar('loss_g_gan_feat', loss_dict['G_GAN_Feat'], step)
                writer.add_scalar('loss_g_vgg', loss_dict['G_VGG'], step)

            step += 1
            iter_end_time = time.time()
            iter_delta_time = iter_end_time - iter_start_time
            step_delta = (step_per_batch - step % step_per_batch) + step_per_batch*(opt.niter + opt.niter_decay-epoch)
            eta = iter_delta_time*step_delta
            eta = str(datetime.timedelta(seconds=int(eta)))
            time_stamp = datetime.datetime.now()
            now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            if MAIN_DEVICE:
                print('{}:{}:[step-{}]--[loss_G-{:.6f}]--[loss_D-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_G, loss_D, eta))

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta and MAIN_DEVICE:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_model(model, 'latest', rank)
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0 and MAIN_DEVICE:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_model(model, 'latest', rank)
            save_model(model, epoch, rank)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            if rank is None:
                model.module.update_fixed_params()
            else:
                model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            if rank is None:
                model.module.update_learning_rate()
            else:
                model.update_learning_rate()


if __name__ == '__main__':
    main()
