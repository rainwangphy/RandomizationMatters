from bat import models
import thop
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
print(Classifier.layerwise_forward(input))

# for name in Classifier.state_dict():
#     print(name)