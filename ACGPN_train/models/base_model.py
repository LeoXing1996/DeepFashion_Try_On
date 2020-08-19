### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
import sys


class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt, rank=None):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if opt.name is not None:  # --> train with exp namd
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        else:  # --> test mode may be no name, but self.save_dir is not used as well
            self.save_dir = opt.checkpoints_dir
        # add code for DDP
        self.rank = rank
        # if rank:
        #     torch.cuda.set_device(rank)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        if self.rank is None or self.rank == 0:
            save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            save_path = os.path.join(self.save_dir, save_filename)
            if hasattr(network, 'module'):
                torch.save(network.module.state_dict(), save_path)
            else:
                torch.save(network.state_dict(), save_path)
        # if len(gpu_ids) and torch.cuda.is_available():
        #     network.cuda()

    def load_dist(self, network, save_file):
        map_location = self.rank
        state_dict = torch.load(save_file, map_location=map_location)
        network.load_state_dict(state_dict)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        print(save_filename)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        elif self.rank:
            self.load_dist(network, save_path)
        else:
            # import pdb
            # pdb.set_trace()
            state_dict = torch.load(save_path)
            state_dict = self.convert_state_dict(state_dict)
            network.load_state_dict(state_dict)

    def convert_state_dict(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                new_state_dict[k[7:]] = v
            else:
                return state_dict
        return new_state_dict

    def update_learning_rate():
        pass
