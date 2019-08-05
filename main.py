# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:07:46 2019

@author: chxy
"""

import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_loader


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    #torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        #torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        #torch.backends.cudnn.deterministic = True
        
    # instantiate data loaders
    test_data_loader = get_test_loader(
        config.data_dir, config.batch_size, **kwargs
    )
    
    if config.is_train:
        train_data_loader = get_train_loader(
            config.data_dir, config.batch_size,
            config.random_seed, config.shuffle, **kwargs
        )
        data_loader = (train_data_loader, test_data_loader)
    else:
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
