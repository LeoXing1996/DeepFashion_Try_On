from util.evaluation import IncepTionScore, SSIMScore

import os
import os.path as op

import torch
from torch.utils.data import DataLoader
from data.eval_dataset import InceptionDataset, SSIMDataset
from options.train_options import TrainOptions
from options.test_options import TestOptions


opt = TestOptions().parse()

assert opt.eval_ssim or opt.eval_inception, "At least given an evaluation option"

# Init Evaluator
if opt.eval_ssim:
    dataset = SSIMDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=opt.nThreads)
    eval_model = SSIMScore(dataloader, opt)
elif opt.eval_inception:
    dataset = InceptionDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, num_workers=opt.nThreads)
    eval_model = IncepTionScore(dataloader, opt)

with torch.no_grad():
    res_dict = eval_model.eval_dataset()

save_dir = op.join('sample', opt.which_ckpt, 'eval')
if opt.remove_old:
    txt_to_remove = 'SSIM.txt' if opt.eval_ssim else 'Inception.txt'
    os.remove(op.join(save_dir, txt_to_remove))
eval_model.save_result(res_dict, save_dir, opt.name)
