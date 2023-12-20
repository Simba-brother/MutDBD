
import random
import numpy as np
from collections import defaultdict
import torch
import timm

import os
import math
from torchvision.transforms import Compose,ToTensor,ToPILImage, Resize
from torch import nn
from copy import deepcopy
import cv2
import sys
sys.path.append("./")
from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import get_dict_state
from codes.datasets.cifar10.attacks import Blended_resnet18_nopretrain_32_32_3 

def test_1():
    random.seed(0)
    for _ in range(20):
        print(random.randint(0,100))

def test_2():
    data = np.random.normal(loc=0.0, scale=1.0, size=8)
    print(data)

def test_3():
    data = torch.arange(0,9).view(3,3)
    
    b = data.get([0,1])
    print(b)

def test_timm():
    print(len(timm.list_models("*"))) 

def test_4():
    num_epochs=300
    num_epoch_repeat = num_epochs//2
    print(num_epoch_repeat)

def test_5():
   device = torch.device('cuda:0')
   print(device)

def test_6():
    print(torch.cuda.is_available())
    print(torch.__version__)

def test_7():
    a = [0,1,2,3]
    a.insert(-2,4)
    print(a)

def test_8():
    a = 9.6
    b = 9.6
    c = torch.tensor(b)
    
def test_9():
    a = [1.1]*10
    a = torch.tensor(a)
    print(a)

def entropy_test(data):
    """
    计算信息熵
    :param data: 数据集
    :return: 信息熵
    """
    length = len(data)
    counter = {}
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent

def test_10():
    data = map(lambda i, j: i + j, [1,2,3], [4,5,6])
    print(data)
    print("fadf")

def test_11():
    poisoned_target_transform = Compose([])
    poisoned_target_transform.transforms.insert(0,ModifyTarget(1))
    print("fda")    

def test_12():
    img = cv2.imread("test/plot_images/1.png")
    # img = np.random.randn(224,224,3)
    transform_test = Compose([
        ToCHW(),
        ToPILImage(), # H,W,C
        Resize((224,224)), # H,W,C
        ToTensor()
    ])
    img_1 = transform_test.transforms[0](img)
    img_2 = transform_test.transforms[1](img_1)
    img_3 = transform_test.transforms[2](img_2)
    img_4 = transform_test.transforms[3](img_3)
    print("")

def test_13():
    torch.tensor([150,180]) <= 200
    print("jlkfajkldf")

def test_14():
    a = math.ceil(4.6)
    print("fka;l")

class ToCHW:
    def __init__(self) -> None:
        pass
    def __call__(self, img):
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array) # H,W,C
        ndim = img_tensor.ndim
        if ndim == 3:
            # 1 张图片
            if img_tensor.shape[0] != 3:
                img_tensor = img_tensor.permute(2,0,1)
        elif ndim == 4:
            if img_tensor.shape[1] != 3:
                img_tensor = img_tensor.permute(2,0,1)
        else:
            raise TypeError("输入既不是4维也不是3维度")
        return img_tensor
    
class ModifyTarget:
    def __init__(self, target_label):
        self.target_label = target_label

    def __call__(self, target_label):
       
         return self.target_label
    
def test15():
    x_list = np.random.permutation(20)
    y_list = np.random.permutation([1,4,5,8,10])
    print(y_list,"\n")
    print("jflakjfla", "\n")
    print(x_list)

def test16():
    with torch.no_grad():
        conv_layer = torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size = 3)
        print(conv_layer.weight.shape)
        weight = conv_layer.weight
        weight[0,0,0,0] = torch.tensor(999.0,requires_grad=True)
        print(conv_layer.weight)
def test17():
    x = [1,2,3,4,5]
    y  = np.random.permutation(x)
    print("fjal")
    
def test18():
    disturb_array = np.random.normal(scale=7, size=100) 
    print(disturb_array.std())
    print()
def test19():
    get_dict_state()
    Blended_resnet18_nopretrain_32_32_3.get_dict_state()
    print()
def test20():

    a = [2] + [1]*(3-1)
    print(a)
def test21():
    # 定义原始列表
    original_list = [5, 2, 8, 1]
    
    # 使用sorted函数进行排序并获取相应的索引
    sorted_indices = sorted(range(len(original_list)), key=lambda x: original_list[x])
    
    print("排序后的索引为：", sorted_indices)
if __name__ == "__main__":
    # test_timm()
    # print(entropy_test([5,1,1,4]))
    # test_13()
    # test15()
    # test16()
    # test19()
    test21()
    pass