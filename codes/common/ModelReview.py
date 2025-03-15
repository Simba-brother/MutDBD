'''
查看模型的结构类
'''
from torchviz import make_dot
import torch
class ModelReview(object):
    def __init__(self):
        self.model = None
    def set_model(self, model):
        self.model = model
    def get_model(self, model):
        return self.model
    def simple_print(self):
        print(self.model)
    def make_dot(self, name:str):
        x = torch.randn(1, 3, 32, 32)
        y = self.model(x)
        make_dot(y, params=dict(self.model.named_parameters())).render(name)
    def see_layers(self):
        model = self.model
        layers = [module for module in model.modules()]
        print(f"总共层数:{len(layers)}")
        print("="*20)
        conv2d_num = 0
        for layer in layers:
            # print(layer,"\n")
            if isinstance(layer, torch.nn.Conv2d):
                conv2d_num += 1
                print(layer.out_channels)
        print("总共卷积层数",conv2d_num)
        print("="*20)
        print("="*20)
        linear_num = 0
        for layer in layers:
            # print(layer,"\n")
            if isinstance(layer, torch.nn.Linear):
                linear_num += 1
                print("in",layer.in_features)
                print("out",layer.out_features)
        print("全连接层数",linear_num)
        print("="*20)