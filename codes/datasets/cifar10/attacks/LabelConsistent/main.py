'''
This is the test code of poisoned training under LabelConsistent.
'''

import sys
sys.path.append("./")
import os
import copy
import os.path as osp
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip,Normalize
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from codes import core
from codes import config
import setproctitle
from codes.core.models.resnet import ResNet
from codes.datasets.cifar10.models.vgg import VGG
from codes.scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset
from codes.common.time_handler import get_formattedDateTime

def _seed_worker(worker_id):
    worker_seed =0
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(model_name):
    model = None
    if model_name == "ResNet18":
        model = ResNet(18,num_classes=10)
    elif model_name == "VGG19":
        model = VGG("VGG19")
    return model

def get_dataset():
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = DatasetFolder(
        root=os.path.join(config.CIFAR10_dataset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = DatasetFolder(
        root=os.path.join(config.CIFAR10_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,testset

def get_trigger():
    # 图片四角白点
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3,:3] = 1.0
    weight[:3,-3:] = 1.0
    weight[-3:,:3] = 1.0
    weight[-3:,-3:] = 1.0
    return pattern,weight

def get_attacker(trainset,testset,victim_model,attack_class,poisoned_rate,
                 adv_model,adv_dataset_dir):

    pattern,weight = get_trigger()
    eps = 8 # Maximum perturbation for PGD adversarial attack. Default: 8.
    alpha = 1.5 # Step size for PGD adversarial attack. Default: 1.5.
    steps = 100 # Number of steps for PGD adversarial attack. Default: 100.
    max_pixel = 255
    attacker = core.LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=victim_model,
        adv_model=adv_model,
        adv_dataset_dir=adv_dataset_dir,# os.path.join(exp_root_dir,"ATTACK", dataset_name, model_name, attack_name, "adv_dataset", f"eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}"),
        loss=nn.CrossEntropyLoss(),
        y_target=attack_class,
        poisoned_rate=poisoned_rate,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    return attacker


def bengin_main(model,trainset,testset):
    poisoned_rate = 0
    adv_model = None
    adv_dataset_dir = None
    attacker = get_attacker(trainset,testset,model,target_class,poisoned_rate,
                            adv_model,adv_dataset_dir)
    attacker.train()
    print("save_path:", os.path.join(attacker.work_dir, "best_model.pth"))
    print("END")
    return attacker.best_model

def attack_main(model,trainset,testset):    
    poisoned_rate = 0.1
    adv_model = copy.deepcopy(model)
    benign_state_dict_path = os.path.join(exp_root_dir,"ATTACK",dataset_name, model_name, attack_name, "benign_train_2025-07-16_17:35:57", "best_model.pth")
    benign_state_dict = torch.load(benign_state_dict_path, map_location="cpu")
    adv_model.load_state_dict(benign_state_dict)
    adv_dataset_dir = os.path.join(exp_root_dir,"ATTACK", dataset_name, model_name, attack_name, "adv_dataset")
    attacker = get_attacker(trainset,testset,model,target_class,poisoned_rate,
                            adv_model,adv_dataset_dir)
    attacker.train()

    print("LC攻击结束,开始保存攻击数据")
    backdoor_model = attacker.best_model
    bd_res = {}
    poisoned_testset = attacker.poisoned_test_dataset
    poisoned_ids = attacker.poisoned_set
    bd_res["backdoor_model"] = backdoor_model
    bd_res["poisoned_ids"] = poisoned_ids
    bd_res["poisoned_testset"] = poisoned_testset
    save_path = os.path.join(
        config.exp_root_dir, "ATTACK",
        dataset_name, model_name, attack_name,
        "backdoor_data.pth")
    torch.save(bd_res, save_path)
    print(f"攻击结果保存到:{save_path}")
    print("END")
    return bd_res

def main(isbenign):
    # 获得受害模型
    victim_model = get_model(model_name)
    # 获得数据集
    trainset,testset = get_dataset()
    if isbenign:
        benign_model = bengin_main(victim_model,trainset,testset)
    else:
        bd_res = attack_main(victim_model,trainset,testset)

if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "CIFAR10"
    attack_name = "LabelConsistent"
    model_name = "VGG19"
    gpu_id = 0
    target_class = 3
    global_seed = 0
    torch.manual_seed(global_seed) # cpu随机数种子
    deterministic = True
    is_benign = False
    experiment_name = "benign_train" if is_benign else "attack_train"
    schedule = {
        'device': f'cuda:{gpu_id}',

        'benign_training': is_benign,
        'batch_size': 128,
        'num_workers': 4,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': osp.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
        'experiment_name': experiment_name
    }
    main(is_benign)




