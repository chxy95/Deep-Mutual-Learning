# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:12:10 2019

@author: chxy
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms

def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    train iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: train set iterator.
    """

    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 将图像转化为32 * 32
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR100(root=data_dir,
                                transform=trans,
                                download=False,
                                train=True)
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader



def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR100 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    trans = transforms.Compose([
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR100(
        data_dir, train=False, download=False, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
