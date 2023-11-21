import os
import sys
sys.path.append("./")
import random
import copy
import math
import time
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader,Dataset

from codes import utils


attack_method = "WaNet" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet

if attack_method == "BadNets":
    from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "Blended":
    from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "IAD":
    from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import PoisonedTrainDataset, PurePoisonedTrainDataset, PureCleanTrainDataset, PoisonedTestSet, TargetClassCleanTrainDataset,  get_dict_state
elif attack_method == "LabelConsistent":
    from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "Refool":
    from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "WaNet":
    from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state



origin_dict_state = get_dict_state()
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
# 本脚本全局变量
# 待变异的后门模型
back_door_model = origin_dict_state["backdoor_model"]
clean_testset = origin_dict_state["clean_testset"]
poisoned_testset = origin_dict_state["poisoned_testset"]
pureCleanTrainDataset = origin_dict_state["pureCleanTrainDataset"]
purePoisonedTrainDataset = origin_dict_state["purePoisonedTrainDataset"]
# mutated model 保存目录
mutation_ratio = 0.05
scale = 1.0
mutation_num = 50
attack_method = "WaNet" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet
work_dir = f"/data/mml/backdoor_detect/experiments/CIFAR10/resnet18_nopretrain_32_32_3/mutates/gf/ratio_{mutation_ratio}_scale_{scale}_num_{mutation_num}/{attack_method}"
# 保存变异模型权重
save_dir = work_dir
utils.create_dir(save_dir)
device = torch.device('cuda:4')

def _seed_worker():
    worker_seed =  666 # torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def multiply(numbers):
    '''
    列表累乘
    '''
    result = 1
    for num in numbers:
        result *= num
    return result

def gen_random_position(weight):
    '''
    从多维矩阵中随机选取一个元素坐标
    '''
    random.seed(1)
    position = []
    for dim_len in tuple(weight.size()):
        position.append(random.randint(0,dim_len-1))
    return position
    
def mutate():
    '''
    对模型进行变异
    '''
    model = back_door_model
    for count in range(mutation_num):
        # copy模型
        model_copy = copy.deepcopy(model)
        # 忽略模型中的一些层
        # ignored_modules = (nn.ReLU, nn.MaxPool2d, nn.Sequential, ResNetBasicBlock, nn.BatchNorm2d)
        # 获得模型层
        # layers = [module for module in model_copy.modules() if not isinstance(module, ignored_modules)]
        layers = [module for module in model_copy.modules()]

        with torch.no_grad():
            # 遍历各层
            for layer in layers[1:]:
                # 层权重
                if hasattr(layer, "weight") is False:
                    continue       
                weight = layer.weight
                # 层权重dimension数目
                ndimension = weight.ndimension()
                # 权重数量
                weight_num = multiply(tuple(weight.size()))
                # 根据变异率确定扰动数量
                disturb_num = math.ceil(weight_num * mutation_ratio)
                # 生成扰动的高斯分布
                disturb_array = np.random.normal(scale=0.05, size=disturb_num) 
                # 遍历每个扰动值
                for i in range(disturb_num):
                    # 获得一个扰动值
                    disturb_value = disturb_array[i]
                    # 随机得到权重矩阵的一个元素位置
                    position = gen_random_position(weight)
                    # 该位置权重+扰动值
                    if isinstance(layer, nn.Conv2d):
                        weight[position[0],position[1],position[2],position[3]] = weight[position[0],position[1],position[2],position[3]] + disturb_value
                    elif isinstance(layer, nn.Linear):
                        weight[position[0],position[1]] = weight[position[0],position[1]] + disturb_value
        file_name = f"model_mutated_{count+1}.pth"
        save_path = os.path.join(save_dir, file_name)
        torch.save(model_copy.state_dict(), save_path)
        print(f"变异模型:{file_name}保存成功, 保存位置:{save_path}")
    print("mutate_model() success")

def eval(m_i, testset):
    # 得到模型结构
    model = back_door_model
    # 加载backdoor weights
    state_dict = torch.load(os.path.join(work_dir, f"model_mutated_{m_i}.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    total_num = len(testset)
    batch_size =128
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    # 评估开始时间
    start = time.time()
    model.to(device)
    model.eval()  # put network in train mode for Dropout and Batch Normalization
    acc = torch.tensor(0., device=device) # 攻击成功率
    correct_num = 0 # 攻击成功数量
    with torch.no_grad():
        for X, Y in testset_loader:
            X = X.to(device)
            Y = Y.to(device)
            preds = model(X)
            correct_num += (torch.argmax(preds, dim=1) == Y).sum()
    acc = correct_num/total_num
    acc = round(acc.item(),3)
    end = time.time()
    print("acc:",acc)
    print(f'Total eval time: {end-start:.1f} seconds')
    print("eval() finished")
    return acc


if __name__ == "__main__":
    # mutate()
    asr_list = []
    acc_list = []
    for m_i in range(mutation_num):
        asr = eval(m_i+1, purePoisonedTrainDataset)
        acc = eval(m_i+1, pureCleanTrainDataset)
        asr_list.append(asr)
        acc_list.append(acc)
    print(asr_list)
    print(f"asr mean:{np.mean(asr_list)}")
    print(acc_list)
    print(f"acc mean:{np.mean(acc_list)}")
    pass