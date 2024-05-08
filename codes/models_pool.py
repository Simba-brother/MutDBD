import os
import torch
from torchvision.models import resnet18,vgg19,densenet121
import torch.nn as nn
from codes import config

class ModelPool(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_resnet18(self):
        if self.dataset_name == "ImageNet":
            model_path = os.path.join(config.exp_root_dir, self.dataset_name, "resnet18.pth")
            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                model = resnet18(pretrained = True)
                # 冻结预训练模型中所有参数的梯度
                for param in model.parameters():
                    param.requires_grad = False
                # 修改最后一个全连接层的输出类别数量
                num_classes = 30  # 假设我们要改变分类数量为30
                fc_features = model.fc.in_features
                model.fc = nn.Linear(fc_features, num_classes)
                torch.save(model, os.path.join(config.exp_root_dir,self.dataset_name, "resnet18.pth"))
            return model
        
    def get_vgg19(self):
        if self.dataset_name == "ImageNet":
            model_path = os.path.join(config.exp_root_dir,self.dataset_name, "vgg19.pth")
            if os.path.exists(model_path):
                model = torch.load(model_path)
                return model
            else:
                model = vgg19(pretrained = True)
                num_classes = 30
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                torch.save(model, os.path.join(config.exp_root_dir,self.dataset_name, "vgg19.pth"))
            return model
        
    def get_densnet(self):
        if self.dataset_name == "ImageNet":
            model_path = os.path.join(config.exp_root_dir,self.dataset_name, "densnet.pth")
            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                model = densenet121(pretrained = True)
                num_classes = 30
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                torch.save(model, os.path.join(config.exp_root_dir,self.dataset_name, "densnet.pth"))
            return model
        