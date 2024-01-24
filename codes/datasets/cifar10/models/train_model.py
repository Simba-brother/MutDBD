
import math
import time
import os
import sys
sys.path.append("./")
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize

from vgg import VGG
# from resnet18_32_32_3 import ResNet
from codes.core.models.resnet import ResNet
from codes.tools.model_train import ModelTrain
from codes import config

exp_root_dir = config.exp_root_dir

def train_cifar10_vgg19():
    model = VGG('VGG19')
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
    batch_size = 128
    epochs = 200
    device = torch.device("cuda:0")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)    
    work_dir = "/data/mml/backdoor_detect/experiments/CIFAR10/vgg19/clean"
    model_train = ModelTrain(model, transform_train, transform_test, trainset, testset, batch_size, epochs, device, loss_fn, optimizer, work_dir, scheduler)
    model_train.train()


def train_cifar10_resnet18():
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    
    model = ResNet(18)
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    transform_test = Compose([
                ToPILImage(),
                Resize((32, 32)),
                ToTensor()
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
    batch_size = 128
    epochs = 200
    device = torch.device("cuda:0")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)   

    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, "clean")

    model_train = ModelTrain(model, transform_train, transform_test, trainset, testset, batch_size, epochs, device, loss_fn, optimizer, save_dir, scheduler)
    model_train.train()
    # 训练好了评估
    model.load_state_dict(torch.load(os.path.join(save_dir,"best_model.pth"), map_location="cpu"))
    model_train = ModelTrain(model, transform_train, transform_test, trainset, testset, batch_size, epochs, device, loss_fn, optimizer, save_dir, scheduler)
    testset_acc = model_train.test()
    print("testset_acc:",testset_acc)

if __name__ == "__main__":
    # train_cifar10_vgg19()
    train_cifar10_resnet18()