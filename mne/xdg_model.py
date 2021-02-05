# from bat import models
# import thop
import torch
from mne.xdg_resnet import XdG_Wide_ResNet

# Classifier = models.Wide_ResNet(28, 10, 0.3, num_classes=10)
#
# # def averaged_adversarial_training():
# #     return 0
#
# print(Classifier)
# input = torch.randn(1, 3, 32, 32)
# flops, params = thop.profile(Classifier,
#                              inputs=(input,))
# print(flops)
# print(params)
#
# for name in Classifier.state_dict():
#     print(name)

# torch.save(Classifier.state_dict(), "class.pt")


# def gen_mask_list(Classifier, mask_rate=0.8):
#     mask_dict = {'conv1': torch.ones(Classifier.activation_list.get('conv1'))}
#     # for stride in self.activation_list['layer1']:
#     for i in range(3):
#         layer = {}
#         for j, block in enumerate(Classifier.activation_list['layer%d' % (i + 1)]):
#             layer_list = []
#             for k in range(len(block)):
#                 mask = torch.ones(block[k])
#                 layer_list.append(mask)
#             layer['block%d' % j] = layer_list
#         mask_dict['layer%d' % (i + 1)] = layer
#     return mask_dict


Classifier = XdG_Wide_ResNet(28, 10, 0.3, num_classes=10)
# print(Classifier)
input = torch.randn(1, 3, 32, 32)
# flops, params = thop.profile(Classifier,
#                              inputs=(input,))
# print(flops)
# print(params)
#
# print(Classifier.activation_list)
print(Classifier(input))
Classifier.gen_mask_list()
print(Classifier(input, Classifier.mask_channel_list[0]))
# print(Classifier.layerwise_forward(input))
# print(len(Classifier.layer1))
# print(len(Classifier.layer2))
# print(len(Classifier.layer2))
# for name in Classifier.state_dict():
#     print(name)

# output_tensor = torch.ones(1, 16, 32, 32)
# masks = torch.ones(16)
# for i in range(16):
#     if torch.rand(1) < 0.5:
#         masks[i] = 0.0
# print(masks)
# masks_tensor = torch.ones_like(output_tensor)
# for i in range(3):
#     masks_tensor[:, i, :, :] = masks[i]
# print(masks_tensor)
