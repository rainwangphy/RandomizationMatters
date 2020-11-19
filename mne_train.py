from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json

import torch
from torch.utils.data import DataLoader as tDataLoader
from torchvision import datasets, transforms

from dataset.DataLoader import DataLoader
from dataset.cifar_dataset import CIFAR10, CIFAR100
from bat import models
from mne import AveragedAttack

import thop

# get the config file for default values
with open('mne/config_mne.json') as config_file:
    config = json.load(config_file)

config['save_dir'] = config['save_dir']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.cuda.device(2):
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
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # with torch.cuda.device(3):
    Classifier_1 = models.Wide_ResNet(28, 10, 0.3, config['number_of_class']).to(device)
    print(Classifier_1)
    # Classifier_2 = models.Wide_ResNet(28, 10, 0.3, config['number_of_class']).to(device)
    # print(Classifier_2)
    #
    # Classifier_list = [Classifier_1, Classifier_2]
    # dis_list = [0.3, 0.7]
    # print(torch.cuda.current_device())
    # attack = AveragedAttack.AveragedPGDAttack(estimator_list=Classifier_list,
    #                                           distribution_list=dis_list,
    #                                           eps=config['eps'] / 255,
    #                                           eps_iter=config['eps_iter'] / 255,
    #                                           nb_iter=2,
    #                                           rand_init=config['rand_init'],
    #                                           clip_min=config['clip_min'],
    #                                           clip_max=config['clip_max'])
    #
    # print(attack)
    # # torch.cuda.set_device(1)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        print(inputs.size())
        flops, paras = thop.profile(Classifier_1, inputs=(inputs,))
        print(flops / 128)
        print(paras)
        # print(Classifier_1(inputs))
        #
        # print(type(inputs))
        # # inputs = attack.perturb(inputs, targets)
        # print(type(inputs))
        break
