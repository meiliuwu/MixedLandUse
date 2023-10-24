import logging
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data_dir = '/home/bpeng/meiliu/landuse/' + args.dataset + '/'
    print(args.dataset)
    print(data_dir)

    # e.g., args.dataset = 'baseline_data_5shot'
    train_dir = os.path.join(data_dir, 'train/')
    test_dir = os.path.join(data_dir, 'test/')
    print(train_dir)

    # data_transform = transforms.Compose([transforms.Resize(224), 
    #                                      transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=transform_train)
    test_data = datasets.ImageFolder(test_dir, transform=transform_test)
    # if args.dataset == "cifar10":
    #     trainset = datasets.CIFAR10(root="./data",
    #                                 train=True,
    #                                 download=True,
    #                                 transform=transform_train)
    #     testset = datasets.CIFAR10(root="./data",
    #                                train=False,
    #                                download=True,
    #                                transform=transform_test) if args.local_rank in [-1, 0] else None

    # else:
    #     trainset = datasets.CIFAR100(root="./data",
    #                                  train=True,
    #                                  download=True,
    #                                  transform=transform_train)
    #     testset = datasets.CIFAR100(root="./data",
    #                                 train=False,
    #                                 download=True,
    #                                 transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    test_sampler = SequentialSampler(test_data)
    
    train_loader = DataLoader(train_data,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if test_data is not None else None

    # prepare data loaders
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, 
    #                                            num_workers=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, 
    #                                           num_workers=4, shuffle=True)
    return train_loader, test_loader
