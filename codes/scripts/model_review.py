import sys
sys.path.append("./")
import torch
from codes.ModelReview import ModelReview
from codes.datasets.cifar10.models.resnet18_32_32_3 import ResNet
from codes.datasets.cifar10.models.vgg import VGG

resnet18 = ResNet(18)
vgg19 = VGG("VGG19")

mr = ModelReview()
mr.set_model(vgg19)
# mr.make_dot("vgg19")
mr.simple_print()

# model = resnet18
# layers = [module for module in model.modules()]
# target_layer = layers[-4]
# weight = target_layer.weight
# print(weight.shape)
# x = torch.randn(1, 3, 32, 32)
# y = model(x)
