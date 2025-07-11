import torch
from torch.utils.data.distributed import DistributedSampler


def get_dataloaders(args, datasets, mini_batch_size, collate_fn=None):
    dataloaders = {}
    for key, dataset in datasets.items():
        if args.use_ddp:
            shuffle = False
            sampler = DistributedSampler(dataset)
        else:
            shuffle = "train" in key
            sampler = None

        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=mini_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        dataloaders[key] = dl
    return dataloaders
