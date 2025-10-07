# CIFAR10,GTSRB
from models.resnet import ResNet # model = ResNet(num=18,num_classes=10)
from models.vgg import VGG  # model = VGG("VGG19")
from models.densenet import densenet_cifar,DenseNet121 # model = densenet_cifar()
# ImageNet2012_subset
import torch.nn as nn
from torchvision.models import resnet18,vgg19,densenet121,VGG19_Weights,ResNet18_Weights,DenseNet121_Weights
def get_model(dataset_name, model_name):
    if dataset_name == "CIFAR10":
        num_classes = 10
        if model_name == "ResNet18":
            model = ResNet(num=18,num_classes=num_classes)
        elif model_name == "VGG19":
            model = VGG("VGG19", num_classes)
        elif model_name == "DenseNet":
            model = densenet_cifar()
    elif dataset_name == "GTSRB":
        # victim model
        num_classes = 43
        if model_name == "ResNet18":
            model = ResNet(num=18, num_classes=num_classes)
        elif model_name == "VGG19":
            model = VGG("VGG19", num_classes)
        elif model_name == "DenseNet":
            model = DenseNet121(num_classes)
    elif dataset_name == "ImageNet2012_subset":
        num_classes = 30
        if model_name == "ResNet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, num_classes)
        elif model_name == "VGG19":
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            # # 冻结其他层
            # for param in model.parameters():
            #     param.requires_grad = False
            # # 开启头层
            # for param in model.classifier[-1].parameters():
            #     param.requires_grad = True
        elif model_name == "DenseNet":
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    return model
