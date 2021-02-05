import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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
            mask_tensor = torch.ones_like(out, requires_grad=False)
            for i in range(len(mask[0])):
                mask_tensor[:, i, :, :] = torch.tensor(mask[0][i])
            out = mask_tensor * out
            out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=0.1))
            out += self.shortcut(x)
            mask_tensor = torch.ones_like(out, requires_grad=False)
            for i in range(len(mask[1])):
                mask_tensor[:, i, :, :] = torch.tensor(mask[1][i])
            out = mask_tensor * out

        return out


class XdG_Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(XdG_Wide_ResNet, self).__init__()
        self.in_planes = 16

        self.best_acc = -1
        self.best_accuracy_under_attack = -1

        self.mask_channel_list = []
        self.freeze_channel_list = []

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        self.n = (depth - 4) // 6
        k = widen_factor

        self.nStages = [16, 16 * k, 32 * k, 64 * k]
        self.dropout_rate = dropout_rate

        self.conv1 = conv3x3(3, self.nStages[0])
        self.layer1, output1 = self._wide_layer(xdg_wide_basic, self.nStages[1], self.n, dropout_rate, stride=1)
        self.layer2, output2 = self._wide_layer(xdg_wide_basic, self.nStages[2], self.n, dropout_rate, stride=2)
        self.layer3, output3 = self._wide_layer(xdg_wide_basic, self.nStages[3], self.n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nStages[3], momentum=0.9)
        self.linear = nn.Linear(self.nStages[3], num_classes)

        self.activation_list = {
            'conv1': self.nStages[0],
            'layer1': output1,
            'layer2': output2,
            'layer3': output3
        }

    def freeze_channel(self):
        freeze_channel_dict = {}
        self.freeze_channel_list.append(freeze_channel_dict)

    # TODO: generate the mask for each layer and each task, i.e., XdG
    def gen_mask_list(self, mask_rate=0.8):
        mask_dict = {}

        mask_dict['conv1'] = [(1.0 if torch.rand(1) > mask_rate else 0) for i in
                              range(self.activation_list.get('conv1'))]
        # for stride in self.activation_list['layer1']:
        for i in range(3):
            layer = {}
            for j, block in enumerate(self.activation_list['layer%d' % (i + 1)]):
                layer_list = []
                for k in range(len(block)):
                    mask = [(1.0 if torch.rand(1) > mask_rate else 0) for i in range(block[k])]
                    layer_list.append(mask)
                layer['block%d' % j] = layer_list
            mask_dict['layer%d' % (i + 1)] = layer

        self.mask_channel_list.append(mask_dict)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        output_number = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            output_number.append([planes, planes])
            self.in_planes = planes

        return nn.Sequential(*layers), output_number

    def forward(self, x, mask=None):
        out = self.conv1(x)

        if mask is None:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)

        else:
            # out should be timed by the mask
            mask_tensor = torch.ones_like(out, requires_grad=False)
            for i in range(len(mask['conv1'])):
                mask_tensor[:, i, :, :] = torch.tensor(mask['conv1'][i])
            out = mask_tensor * out
            for i in range(len(self.layer1)):
                out = self.layer1[i](out, mask['layer1']['block%d' % i])
            for i in range(len(self.layer2)):
                out = self.layer2[i](out, mask['layer2']['block%d' % i])
            for i in range(len(self.layer3)):
                out = self.layer3[i](out, mask['layer3']['block%d' % i])

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
