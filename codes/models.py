import torch
import torch.nn as nn
from codes.core.models.resnet import ResNet
from torchvision.models import resnet18,vgg19,densenet121
from codes.asd.models import resnet_cifar # ASD开源项目中的模型
from codes.datasets.cifar10.models.vgg import VGG
from codes.datasets.GTSRB.models.vgg import VGG as GTSRB_VGG
from codes.datasets.cifar10.models.densenet import densenet_cifar
from codes.datasets.GTSRB.models.densenet import DenseNet121
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# def get_resnet(num,num_classes):
#     return ResNet(num,num_classes) # ResNet(18,10)

# def get_resnet_cifar(num_classes):
#     return resnet_cifar.get_model(num_classes)

def get_model(dataset_name,model_name):
    if dataset_name == "CIFAR10":
        if model_name == "ResNet18":
            return ResNet(num=18,num_classes=10)
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
    elif dataset_name == "ImageNet2012_subset":
        num_classes = 30
        if model_name == "ResNet18":
            model = resnet18(pretrained = True)
            fc_features = model.fc.in_features
            model.fc = nn.Linear(fc_features, num_classes)
        elif model_name == "VGG19":
            deterministic = False
            model = vgg19(pretrained = True)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name == "DenseNet":
            model = densenet121(pretrained = True)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        return model

if __name__ == "__main__":
    model = get_model("ImageNet2012_subset","DenseNet")
    '''
    if model_name == "ResNet18":
        in_features = model.fc.in_features
        node_str = "flatten"
    elif model_name == "VGG19":
        in_features = model.classifier[-1].in_features
        node_str = "classifier.5"
    elif model_name == "DenseNet":
        in_features = model.classifier.in_features
        node_str = "flatten"
    ''' 
    data = torch.rand(1,3,224,224)
    node_str = "flatten"
    feature_extractor = create_feature_extractor(model, return_nodes=[node_str])
    feature_dic = feature_extractor(data)
    feature = feature_dic[node_str]
    print(model)

        
