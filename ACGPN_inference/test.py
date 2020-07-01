import time
from collections import OrderedDict
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2

# from util.SSIM import SSIM

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


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
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


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


os.makedirs('sample', exist_ok=True)
opt = TrainOptions().parse()

save_dir = os.path.join('sample', opt.name)
img_dir = os.path.join(save_dir, 'img')
summ_dir = os.path.join(save_dir, 'logs')
debug_dir = os.path.join(save_dir, 'debug')

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except FileNotFoundError:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
    opt.nThread = 1
    # import pdb
    # pdb.set_trace()

if not os.path.exists('debug_img'):
    os.makedirs('debug_img')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# Inference images = %d' % dataset_size)

model = create_model(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

step = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        #save_fake = total_steps % opt.display_freq == display_delta
        save_fake = True

        ##add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        #
        # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        ############## Forward Pass ######################
        with torch.no_grad():
            fake_image, real_image, real_cloth, cloth_gt, cloth_warp, \
                fake_label, clothes_mask = \
                model(Variable(data['label'].cuda()), Variable(data['edge'].cuda()),
                      Variable(img_fore.cuda()), Variable(mask_clothes.cuda()),
                      Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()),
                      Variable(data['image'].cuda()), Variable(data['pose'].cuda()),
                      Variable(data['image'].cuda()), Variable(mask_fore.cuda()), data['name'])

        ############## Display results and errors ##########

        ### display output images
        a = generate_label_color(generate_label_plain(fake_label)).float().cuda()
        b = real_cloth.float().cuda()
        c = fake_image.float().cuda()
        d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
        combine = torch.cat([a[0], d[0], b[0], c[0], real_image[0]], 2).squeeze()
        # combine=c[0].squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy()+1) / 2
        fake_img_cv = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        if step % 1 == 0 and not opt.no_img:
            real_image = (cv_img*255).astype(np.uint8)
            fake_rgb = (fake_img_cv*255).astype(np.uint8)

            bgr = cv2.cvtColor(real_image, cv2.COLOR_RGB2BGR)
            fake_bgr = cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2BGR)
            n = str(step)+'.jpg'
            cv2.imwrite(img_dir + '/combine_' + data['name'][0], bgr)
            cv2.imwrite(img_dir + '/' + data['name'][0], fake_bgr)
        step += 1
        print(step)
        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    break

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
