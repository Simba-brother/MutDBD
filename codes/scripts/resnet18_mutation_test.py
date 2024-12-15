import sys
sys.path.append("./")
from collections import defaultdict
import os
import cv2
import joblib

import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision.datasets import DatasetFolder

from codes.ourMethod.modelMutat import ModelMutat
from codes.common.eval_model import EvalModel
from codes import config
from core.models.resnet import ResNet
from codes.tools.draw import draw_line
from utils import create_dir
dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name


model = ResNet(num=18,num_classes=10)
clean_state_dict_path = "/data/mml/backdoor_detect/experiments/CIFAR10/resnet18_nopretrain_32_32_3/clean/best_model.pth"
model.load_state_dict(torch.load(clean_state_dict_path, map_location="cpu"))
device = torch.device("cuda:1")

transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])

trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

cur_dataset = trainset

e = EvalModel(model, cur_dataset, device)
origin_acc = e._eval_acc()

mutation_ratio_list = [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
exp_root_dir = "/data/mml/backdoor_detect/experiments/"

def gf_test():
    acc_list = []
    for mutation_ratio in mutation_ratio_list:
        mean_acc = 0
        for m_i in range(50):
            modelMutat = ModelMutat(model, mutation_ratio)
            mutated_model = modelMutat._gf_mut()
            e = EvalModel(mutated_model, cur_dataset, device)
            acc = e._eval_acc()
            mean_acc += acc
        mean_acc /= 50
        mean_acc = round(mean_acc,3)
        acc_list.append(mean_acc) 
    acc_list.insert(0,origin_acc)
    mutation_ratio_list.insert(0,0)
    # 保存数据
    save_dir = os.path.join(exp_root_dir, "temp_data")
    create_dir(save_dir)
    save_file_name = "acc_list.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(acc_list,save_path)
    # 绘图
    x_list = mutation_ratio_list
    title = "Mutation operator:GF, Model:ResNet18, Dataset:CIFAR10"
    xlabel = "mutation rate"
    save_dir = os.path.join(exp_root_dir, "images/line", dataset_name, model_name, "mutation_test")
    create_dir(save_dir)
    save_file_name = "GF.png"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"Accuracy":acc_list}
    draw_line(x_list,title, xlabel, save_path, **y)
    # 返回数据
    return acc_list

def inverse_test():
    acc_list = []
    for mutation_ratio in mutation_ratio_list:
        mean_acc = 0
        for m_i in range(50):
            modelMutat = ModelMutat(model, mutation_ratio)
            mutated_model = modelMutat._neuron_activation_inverse()
            e = EvalModel(mutated_model, cur_dataset, device)
            acc = e._eval_acc()
            mean_acc += acc
        mean_acc /= 50
        mean_acc = round(mean_acc,3)
        acc_list.append(mean_acc) 
    acc_list.insert(0,origin_acc)
    mutation_ratio_list.insert(0,0)
    # 保存数据
    save_dir = os.path.join(exp_root_dir, "temp_data")
    create_dir(save_dir)
    save_file_name = "acc_list.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(acc_list,save_path)
    # 绘图
    x_list = mutation_ratio_list
    title = "Mutation operator:inverse, Model:ResNet18, Dataset:CIFAR10"
    xlabel = "mutation rate"
    save_dir = os.path.join(exp_root_dir, "images/line", dataset_name, model_name, "mutation_test")
    create_dir(save_dir)
    save_file_name = "inverse.png"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"Accuracy":acc_list}
    draw_line(x_list,title, xlabel, save_path, **y)
    # 返回数据
    return acc_list

def block_test():
    acc_list = []
    for mutation_ratio in mutation_ratio_list:
        mean_acc = 0
        for m_i in range(50):
            modelMutat = ModelMutat(model, mutation_ratio)
            mutated_model = modelMutat._neuron_block()
            e = EvalModel(mutated_model, cur_dataset, device)
            acc = e._eval_acc()
            mean_acc += acc
        mean_acc /= 50
        mean_acc = round(mean_acc,3)
        acc_list.append(mean_acc) 
    acc_list.insert(0,origin_acc)
    mutation_ratio_list.insert(0,0)
    # 保存数据
    save_dir = os.path.join(exp_root_dir, "temp_data")
    create_dir(save_dir)
    save_file_name = "acc_list.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(acc_list,save_path)
    # 绘图
    x_list = mutation_ratio_list
    title = "Mutation operator:block, Model:ResNet18, Dataset:CIFAR10"
    xlabel = "mutation rate"
    save_dir = os.path.join(exp_root_dir, "images/line", dataset_name, model_name, "mutation_test")
    create_dir(save_dir)
    save_file_name = "block.png"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"Accuracy":acc_list}
    draw_line(x_list,title, xlabel, save_path, **y)
    # 返回数据
    return acc_list

def switch_test():
    acc_list = []
    for mutation_ratio in mutation_ratio_list:
        mean_acc = 0
        for m_i in range(50):
            modelMutat = ModelMutat(model, mutation_ratio)
            mutated_model = modelMutat._neuron_switch()
            e = EvalModel(mutated_model, cur_dataset, device)
            acc = e._eval_acc()
            mean_acc += acc
        mean_acc /= 50
        mean_acc = round(mean_acc,3)
        acc_list.append(mean_acc) 
    acc_list.insert(0,origin_acc)
    mutation_ratio_list.insert(0,0)
    # 保存数据
    save_dir = os.path.join(exp_root_dir, "temp_data")
    create_dir(save_dir)
    save_file_name = "acc_list.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(acc_list,save_path)
    # 绘图
    x_list = mutation_ratio_list
    title = "Mutation operator:switch, Model:ResNet18, Dataset:CIFAR10"
    xlabel = "mutation rate"
    save_dir = os.path.join(exp_root_dir, "images/line", dataset_name, model_name, "mutation_test")
    create_dir(save_dir)
    save_file_name = "switch.png"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"Accuracy":acc_list}
    draw_line(x_list,title, xlabel, save_path, **y)
    # 返回数据
    return acc_list

def shuffle_test():
    acc_list = []
    for mutation_ratio in mutation_ratio_list:
        mean_acc = 0
        for m_i in range(50):
            modelMutat = ModelMutat(model, mutation_ratio)
            mutated_model = modelMutat._weight_shuffling()
            e = EvalModel(mutated_model, cur_dataset, device)
            acc = e._eval_acc()
            mean_acc += acc
        mean_acc /= 50
        mean_acc = round(mean_acc,3)
        acc_list.append(mean_acc) 
    acc_list.insert(0,origin_acc)
    mutation_ratio_list.insert(0,0)
    # 保存数据
    save_dir = os.path.join(exp_root_dir, "temp_data")
    create_dir(save_dir)
    save_file_name = "acc_list.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(acc_list,save_path)
    # 绘图
    x_list = mutation_ratio_list
    title = "Mutation operator:shuffle, Model:ResNet18, Dataset:CIFAR10"
    xlabel = "mutation rate"
    save_dir = os.path.join(exp_root_dir, "images/line", dataset_name, model_name, "mutation_test")
    create_dir(save_dir)
    save_file_name = "shuffle.png"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"Accuracy":acc_list}
    draw_line(x_list,title, xlabel, save_path, **y)
    # 返回数据
    return acc_list

if __name__ == "__main__":
    # gf_test()
    # inverse_test()
    # block_test()
    # switch_test()
    shuffle_test()
    pass
