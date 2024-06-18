import config    
from core.models.resnet import ResNet
from codes.asd.models import resnet_cifar

def get_resnet(num,num_classes):
    return ResNet(num,num_classes) # ResNet(18,10)

def get_resnet_cifar(num_classes):
    return resnet_cifar.get_model(num_classes)
