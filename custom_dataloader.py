import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def load_data(opt, normal_classes, train=True, check=False):

    transform = transforms.Compose(
        [
            transforms.Resize(opt.isize),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    if opt.dataset == 'MNIST':
        dataset = MNIST(root='./data/', train=train, download=True, transform=transform)

    normal_img, normal_lb, abnormal_img, abnormal_lb = get_mnist_anomaly(dataset.data, dataset.targets, normal_classes)

    # 일단 train 전용으로
    # if check:
    #     dataset.data = abnormal_img
    #     dataset.targets = abnormal_lb
    # else:
    #     if train:
    #         dataset.data = normal_img
    #         dataset.targets = normal_lb
    #     else:
    #         dataset.data = torch.cat((abnormal_img,normal_img),dim=0)
    #         dataset.targets = torch.cat((abnormal_lb,normal_lb), dim=0)

    dataset.data = normal_img
    dataset.targets = normal_lb
    
    dataloader = DataLoader(dataset, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers,
                            drop_last=False)

    return dataloader

def get_mnist_anomaly(img, lbl, normal_c:list):
    normal_idx = torch.from_numpy(np.where(np.isin(lbl.numpy(),normal_c))[0])
    abnormal_idx = torch.from_numpy(np.where((np.isin(lbl.numpy(), normal_c,invert=True)))[0])

    normal_img = img[normal_idx]
    abnormal_img = img[abnormal_idx]
    normal_lbl = lbl[normal_idx]
    abnormal_lbl = lbl[abnormal_idx]

    for idx,c in enumerate(normal_c):
        c_idx = torch.from_numpy(np.where(normal_lbl.numpy() == c)[0])
        normal_lbl[c_idx] = idx+1

    abnormal_lbl[:] = 0
    
    return normal_img,normal_lbl, abnormal_img, abnormal_lbl