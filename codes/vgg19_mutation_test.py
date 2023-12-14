import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import numpy as np
import cv2

sys.path.append("./")
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision.datasets import DatasetFolder
from codes import config
from torchvision import transforms

dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
# mutation_name = config.mutation_name 
# mutation_name_list =  config.mutation_name_list

from codes.datasets.cifar10.models.vgg import VGG

model = VGG("VGG19")
clean_state_dict_path = "/data/mml/backdoor_detect/experiments/CIFAR10/vgg19/clean/best_model.pth"
model.load_state_dict(torch.load(clean_state_dict_path))
ratio = 1.0
device = torch.device("cuda:1")

transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# 获得数据集
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
origin_test_acc = e._eval_acc()


modelMutat = ModelMutat(model, mutation_ratio=ratio)

def gf_test():
    gf_acc_list = []
    for m_i in range(50):
        mutated_model = modelMutat._gf_mut()
        e = EvalModel(mutated_model, cur_dataset, device)
        acc = e._eval_acc()
        gf_acc_list.append(acc)
    gf_mean_acc = np.mean(gf_acc_list)
    return gf_mean_acc

def inverse_test():
    inverse_acc_list = []
    for m_i in range(50):
        mutated_model = modelMutat._neuron_activation_inverse()
        e = EvalModel(mutated_model, cur_dataset, device)
        acc = e._eval_acc()
        inverse_acc_list.append(acc)
    inverse_mean_acc = np.mean(inverse_acc_list)
    return inverse_mean_acc

def block_test():
    block_acc_list = []
    for m_i in range(50):
        mutated_model = modelMutat._neuron_block()
        e = EvalModel(mutated_model, cur_dataset, device)
        acc = e._eval_acc()
        block_acc_list.append(acc)
    block_mean_acc = np.mean(block_acc_list)
    return block_mean_acc

def switch_test():
    switch_acc_list = []
    for m_i in range(50):
        mutated_model = modelMutat._neuron_switch()
        e = EvalModel(mutated_model, cur_dataset, device)
        acc = e._eval_acc()
        switch_acc_list.append(acc)
    switch_mean_acc = np.mean(switch_acc_list)
    return switch_mean_acc

def shuffle_test():
    shuffle_acc_list = []
    for m_i in range(50):
        mutated_model = modelMutat._weight_shuffling()
        e = EvalModel(mutated_model, cur_dataset, device)
        acc = e._eval_acc()
        shuffle_acc_list.append(acc)
    shuffle_mean_acc = np.mean(shuffle_acc_list)
    return shuffle_mean_acc

if __name__ == "__main__":
    gf_mean_acc = gf_test()
    inverse_mean_acc = inverse_test()
    block_mean_acc = block_test()
    switch_mean_acc = switch_test()
    shuffle_mean_acc = shuffle_test()
    print("ratio:",ratio)
    print("origin_acc:",origin_test_acc)
    print("gf_mean_acc:",gf_mean_acc)
    print("inverse_mean_acc:",inverse_mean_acc)
    print("block_mean_acc:",block_mean_acc)
    print("switch_mean_acc:",switch_mean_acc)
    print("shuffle_mean_acc:",shuffle_mean_acc)