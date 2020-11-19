from bat import models
import thop
import torch
from mne.xdg_resnet import XdG_Wide_ResNet

Classifier = models.Wide_ResNet(28, 10, 0.3, num_classes=10)

# def averaged_adversarial_training():
#     return 0

print(Classifier)
input = torch.randn(1, 3, 32, 32)
flops, params = thop.profile(Classifier,
                             inputs=(input,))

print(Classifier.layer1[0])
