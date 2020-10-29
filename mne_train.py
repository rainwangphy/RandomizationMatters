from __future__ import print_function

import json

import torch
from torch.utils.data import DataLoader as tDataLoader
from torchvision import datasets, transforms

from dataset.DataLoader import DataLoader
from dataset.cifar_dataset import CIFAR10, CIFAR100

# get the config file for default values
with open('mne/config.json') as config_file:
    config = json.load(config_file)

config['save_dir'] = config['save_dir']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading datasets
config['dataroot'] = config['dataroot'] + "/" + config['dataset']

custom_data_class = CIFAR10
original_data_class = datasets.CIFAR10

if config['dataset'] == 'cifar10':
    custom_data_class = CIFAR10
    original_data_class = datasets.CIFAR10
    config['number_of_class'] = 10

elif config['dataset'] == 'cifar100':
    custom_data_class = CIFAR100
    original_data_class = datasets.CIFAR100
    config['number_of_class'] = 100

train_loader = DataLoader(
    custom_data_class(
        root=config['dataroot'],
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    ),
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2,
    drop_last=True
)

test_loader = tDataLoader(
    original_data_class(
        root=config['dataroot'],
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])),
    batch_size=config['test_batch_size'],
    shuffle=False,
    num_workers=2,
    drop_last=True
)
