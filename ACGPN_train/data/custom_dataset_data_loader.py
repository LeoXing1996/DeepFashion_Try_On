import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data import DistributedSampler


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, rank=None):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # specific parameters for DDP
        dist = rank is not None
        sampler = DistributedSampler(self.dataset) if dist else None
        shuffle = False if dist else not opt.serial_batches
        batch_size = opt.batchSize // len(opt.gpu_ids) if dist else opt.batchSize
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(opt.nThreads))
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batchSize,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
