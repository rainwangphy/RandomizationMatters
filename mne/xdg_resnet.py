import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np


# The methods conv3x3, conv_init, wide_basic and the class Wide_Resnet
# have been taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
# We have modified the class Wide_Resnet.

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class xdg_wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(xdg_wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x, mask=None):
        if mask is None:
            out = self.conv1(F.leaky_relu(self.bn1(x), negative_slope=0.1))
            # out = self.dropout(out)
            out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=0.1))
            out += self.shortcut(x)
        else:
            # two masks are considered:
            # 1. the mask just after first conv1
            # 2. the mask after the residual connection
            out = self.conv1(F.leaky_relu(self.bn1(x), negative_slope=0.1))
            # This dropout procedure does not appear in other implementation,
            # here, we instead use a mask to replace the dropout
            # out = self.dropout(out)

            out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=0.1))
            out += self.shortcut(x)

        return out


class XdG_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(XdG_Wide_ResNet, self).__init__()
        self.in_planes = 16

        self.best_acc = -1
        self.best_accuracy_under_attack = -1

        self.mask_list = []

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        self.n = (depth - 4) // 6
        k = widen_factor

        self.nStages = [16, 16 * k, 32 * k, 64 * k]
        self.dropout_rate = dropout_rate

        self.conv1 = conv3x3(3, self.nStages[0])
        self.layer1, output1, stride_number1 = self._wide_layer(xdg_wide_basic, self.nStages[1], self.n, dropout_rate,
                                                                stride=1)
        self.layer2, output2, stride_number2 = self._wide_layer(xdg_wide_basic, self.nStages[2], self.n, dropout_rate,
                                                                stride=2)
        self.layer3, output3, stride_number3 = self._wide_layer(xdg_wide_basic, self.nStages[3], self.n, dropout_rate,
                                                                stride=2)
        self.bn1 = nn.BatchNorm2d(self.nStages[3], momentum=0.9)
        self.linear = nn.Linear(self.nStages[3], num_classes)

        self.activation_list = {
            'conv1': self.nStages[0],
            'layer1': output1,
            'layer2': output2,
            'layer3': output3
        }

        self.layer1_stride = stride_number1
        self.layer2_stride = stride_number2
        self.layer3_stride = stride_number3

    # generate the mask list for each task, i.e., XdG
    def gen_mask_list(self):
        self.mask_list.append(1)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        output_number = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            output_number.append([planes, planes])
            self.in_planes = planes

        return nn.Sequential(*layers), output_number, len(strides)

    def forward(self, x, mask=None):
        out = self.conv1(x)
        # out should be timed by the mask
        # for block in self.layer1:
        #     out = block(out)
        # for block in self.layer2:
        #     out = block(out)
        # for block in self.layer3:
        #     out = block(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.leaky_relu(self.bn1(out), negative_slope=0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def layerwise_forward(self, x, mask=None):
        out = self.conv1(x)
        for i in range(self.layer1_stride):
            out = self.layer1[i](out)
        for i in range(self.layer2_stride):
            out = self.layer2[i](out)
        for i in range(self.layer3_stride):
            out = self.layer3[i](out)
        out = F.leaky_relu(self.bn1(out), negative_slope=0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def updateBestAccuracies(self, accuracy, accuracy_under_attack):
        acc = False
        acc_under_attack = False

        if accuracy > self.best_acc:
            self.best_acc = accuracy
            acc = True
        if accuracy_under_attack > self.best_accuracy_under_attack:
            self.best_accuracy_under_attack = accuracy_under_attack
            acc_under_attack = True

        return acc, acc_under_attack
