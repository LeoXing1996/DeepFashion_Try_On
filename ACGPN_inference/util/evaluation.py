import torch
from torch.nn import functional as F
from torchvision.models.inception import inception_v3

from math import exp
import os.path as op
import numpy as np
from scipy.stats import entropy


class BaseEvaluation:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.N = dataloader.dataset.__len__()
        self.batch_size = dataloader.batch_size
        if len(dataloader) * dataloader.batch_size != self.N and dataloader.drop_last:
            print('Drop_last option is TRUE in dataloader.')
            print('Only {}/{} images can be loaded'.format(dataloader.dataset.__len__(), self.N))
            exit(0)

    def print_func(self, score, step, interval=1, print_output=True):
        if (step+1) % interval == 0 and print_output:
            print('{}/{} score: {}'.format(step, self.N, float(score)))

    @staticmethod
    def check_input(input):
        raise NotImplementedError

    def forward_single(self, *inputs):
        raise NotImplementedError

    def eval_dataset(self, *args, **kwargs):
        raise NotImplementedError


class SSIMScore(BaseEvaluation):
    def __init__(self, dataloader, opt):
        super().__init__(dataloader)
        self.window_size = opt.window_size
        self.size_average = opt.size_average
        self.channel = opt.channel
        self.window = self.create_window(self.window_size, self.channel)

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    @staticmethod
    def check_input(input):
        # 1. 0 <= input <= 1
        # assert input.min() >= 0 and input.max() <= 1
        assert (0 <= input <= 1).all()
        assert input.shape[0] == 1, "Batch size for SSIM evaluation should be 1 !"

    def forward_single(self, img1, img2):
        window, window_size, channel = self.window, self.window_size, self.channel
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def eval_dataset(self, print_output=True):
        ssim_list, step = [], 0
        for data in self.dataloader:
            img_F, img_R = data['img_F'], data['img_R']
            ssim_val = self.forward_single(img_F, img_R)
            ssim_list.append(ssim_val)
            step += 1
            self.print_func(ssim_val, step, print_output=print_output)
        res_dict = {'mean': np.mean(ssim_list),
                    'std': np.std(ssim_list)}
        return res_dict

    def save_result(self, res, path):
        mean_list = res['mean']
        std_list = res['std']
        res_strs = [
            'MEAN  : {}'.format(float(mean_list)),
            'STD   : {}'.format(float(std_list)),
        ]
        with open(op.join(path, 'SSIM.txt'), 'w') as file:
            for r in res_strs:
                print(r)
            file.write('\n'.join(res_strs))


class IncepTionScore(BaseEvaluation):
    def __init__(self, dataloader, opt):
        super().__init__(dataloader)
        self.model = inception_v3(pretrained=True, transform_input=False).cuda()
        self.model.eval()
        self.splits = opt.splits
        self.out_num = opt.out_num

    @staticmethod
    def check_input(img):
        # 1. check input size
        assert list(img.shape[1:]) == [3, 299, 299]
        # 2. check input range
        assert isinstance(img, torch.FloatTensor) or \
             isinstance(img, torch.cuda.FloatTensor)
        return True

    def eval_dataset(self, print_output=True):
        splits, N = self.N, self.splits
        preds = np.zeros((N, self.out_num))
        batch_size = self.batch_size
        print('Start running Inception V3 Preds.....')
        for i, batch in enumerate(self.dataloader):
            data = batch['img']
            data = data.cuda()
            batch_size_i = data.size()[0]
            preds[i*batch_size: i*batch_size + batch_size_i] = self.forward_single(data)
        # Now compute the mean kl-div
        print('Running Finish, Start calculate Inception Score !')
        split_scores = []

        step = 0
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                score = entropy(pyx, py)
                self.print_func(score, step, print_output=print_output)
                # scores.append(entropy(pyx, py))
                scores.append(score)
                step += 1
            split_scores.append(np.exp(np.mean(scores)))
        mean, std = np.mean(split_scores), np.std(split_scores)
        res_dict = {'mean': mean, 'std': std, 'splits': splits}
        return res_dict

    def forward_single(self, img):
        pred = self.model(img)
        return F.softmax(pred, dim=1).cpu().numpy()

    def save_result(self, res, path):
        mean = res['mean']
        std = res['std']
        splits = res['splits']
        res_strs = [
            'SPLITS: {}'.format(float(splits)),
            'MEAN  : {}'.format(float(mean)),
            'STD   : {}'.format(float(std))
        ]
        with open(op.join(path, 'Inception.txt'), 'w') as file:
            for r in res_strs:
                print(r)
            file.write('\n'.join(res_strs))
