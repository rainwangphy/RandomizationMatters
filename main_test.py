from bat import models
import thop
import torch
from mne.xdg_resnet import XdG_Wide_ResNet
#
# Classifier = models.Wide_ResNet(28, 10, 0.3, num_classes=10)
#
# # def averaged_adversarial_training():
# #     return 0
#
# print(Classifier)
# # input = torch.randn(1, 3, 32, 32)
# # flops, params = thop.profile(Classifier,
# #                              inputs=(input,))
# #
# # print(Classifier.layer1[0])

import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
#
#
# # The methods conv3x3, conv_init, wide_basic and the class Wide_Resnet
# # have been taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
# # We have modified the class Wide_Resnet.
#
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
#
#
# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=np.sqrt(2))
#         init.constant(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0)
#
#
# conv1 = conv3x3(3, 16)
#
# input_tensor = torch.randn(1, 3, 32, 32)
#
# out = conv1(input_tensor)
# print(out)
# do = nn.Dropout(p=0.2)
# out = do(out)
# print(out)

b = 1 if torch.rand(1) > 0.5 else 0
print(b)
