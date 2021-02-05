from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import json

import torch
from torch.utils.data import DataLoader as tDataLoader
from torchvision import datasets, transforms

from bat.mixture import Mixture_of_Classifier
from dataset.DataLoader import DataLoader
from dataset.cifar_dataset import CIFAR10, CIFAR100

# get the config file for default values
with open('bat/config.json') as config_file:
    config = json.load(config_file)

config['save_dir'] = config['save_dir']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading datasets
#

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
    num_workers=0,
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
    num_workers=0,
    drop_last=True
)

# Build the mixture class

MC = Mixture_of_Classifier(train_loader=train_loader,
                           test_loader=test_loader,
                           device=device,
                           config=config)

# Variable for printing
level = 0

# Boolean variable that help us to know if we have enough saved classifier to load.
completeLoading = True
if config['load']:
    completeLoading = MC.load(top_accuracy_under_attack=True, level=level + 1)

# If we have loaded less classifiers than config['number_of_models'] or if we haven't loaded anything,
# we run the method to construct the classifiers until having config['number_of_models'] classifers
if config['load'] is False or completeLoading is False:
    MC.boosting(level=level + 1)
