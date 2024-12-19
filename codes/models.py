import torch.nn as nn
from codes.core.models.resnet import ResNet
from torchvision.models import resnet18,vgg19,densenet121
from codes.asd.models import resnet_cifar # ASD开源项目中的模型
from codes.attack.cifar10.models.vgg import VGG
from codes.attack.GTSRB.models.vgg import VGG as GTSRB_VGG
from codes.attack.cifar10.models.densenet import densenet_cifar
from codes.attack.GTSRB.models.densenet import DenseNet121


# def get_resnet(num,num_classes):
#     return ResNet(num,num_classes) # ResNet(18,10)

# def get_resnet_cifar(num_classes):
#     return resnet_cifar.get_model(num_classes)

def get_model(dataset_name,model_name):
    if dataset_name == "CIFAR10":
        if model_name == "ResNet18":
            return ResNet(num=18,num_classes=10),
        elif model_name == "VGG19":
            return VGG("VGG19")
        elif model_name == "DenseNet":
            return densenet_cifar()
    elif dataset_name == "GTSRB":
        if model_name == "ResNet18":
            return ResNet(num=18,num_classes=43)
        elif model_name == "VGG19":
            return GTSRB_VGG("VGG19", 43)
        elif model_name == "DenseNet":
            return DenseNet121(43)
    elif dataset_name == "ImageNet":
        if model_name == "ResNet18":
            model = resnet18(pretrained = True)
            # 修改最后一个全连接层的输出类别数量
            num_classes = 30  # 假设我们要改变分类数量为30
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, num_classes)
            return model
        elif model_name == "VGG19":
            # victim model
            model = vgg19(pretrained = True)
            # 冻结预训练模型中所有参数的梯度
            for param in model.parameters():
                param.requires_grad = False
            num_classes = 30
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model
        elif model_name == "DenseNet":
            # victim model
            model = densenet121(pretrained = True)
            # 冻结预训练模型中所有参数的梯度
            # for param in model.parameters():
            #     param.requires_grad = False
            # 冻结部分参数
            for module in model.features[0:6]:
                for param in module.parameters():
                    param.requires_grad = False

            # 修改最后一个全连接层的输出类别数量
            num_classes = 30  # 假设我们要改变分类数量为30
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model
