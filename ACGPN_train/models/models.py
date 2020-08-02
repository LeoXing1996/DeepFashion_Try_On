### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch


def create_model(opt):
    if opt.model == 'pix2pixHD':
        # from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        from .pix2pixHD_model import Pix2PixHDModel
        if opt.isTrain:
            model = Pix2PixHDModel()
            #ipdb.set_trace()
        # else:
        #     model = InferenceModel()

    model.initialize(opt, opt.local_rank)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    # if opt.dist model would apply dist itself
    if opt.isTrain and len(opt.gpu_ids) and (opt.local_rank is None):
        print('use DP, not good')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
