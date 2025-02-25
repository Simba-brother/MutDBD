import os
import cv2
import random

import setproctitle
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, ToPILImage, Resize, RandomResizedCrop, Normalize, CenterCrop

from torchvision.models import resnet18,vgg19,densenet121
from codes.core.attacks import BadNets
from codes import config
from codes.datasets.ImageNet.attacks.BadNets.utils import create_backdoor_data
from codes.datasets.utils import eval_backdoor,update_backdoor_data
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset
from codes.scripts.dataset_constructor import ExtractDataset

global_seed = config.random_seed
deterministic = True
# cpu种子
torch.manual_seed(global_seed)

exp_root_dir = config.exp_root_dir
dataset_name = "ImageNet2012_subset"
model_name = "ResNet18"
attack_name = "BadNets"

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

def _seed_worker(worker_id):
    np.random.seed(global_seed)
    random.seed(global_seed)

'''
原始的
# 训练集transform    
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor() # CHW
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor()
])
'''
# 训练集transform    
transform_train = Compose([
    ToPILImage(),
    RandomResizedCrop(size=224), 
    ToTensor()
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor()
])

# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir, "train"),
    loader=cv2.imread, # ndarray (H,W,C)
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir, "test"),
    loader=cv2.imread, # ndarray(shape:HWC)
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

# backdoor pattern
pattern = torch.zeros((224, 224), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((224, 224), dtype=torch.float32)
weight[-3:, -3:] = 1.0

badnets = BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index= -1,
    poisoned_transform_test_index= -1,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': f'cuda:{config.gpu_id}',
    
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 150], # epoch区间 (150,180)

    'epochs': 20, # 默认: 200

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}

def attack():
    # 攻击
    badnets.train(schedule)
    # 工作dir
    work_dir = badnets.work_dir
    # 获得backdoor model weights
    backdoor_model = badnets.best_model
    # clean testset
    clean_testset = testset
    # poisoned testset
    poisoned_testset = badnets.poisoned_test_dataset
    # poisoned trainset
    poisoned_trainset = badnets.poisoned_train_dataset
    # poisoned_ids
    poisoned_ids = poisoned_trainset.poisoned_set


    dict_state = {}
    dict_state["backdoor_model"] = backdoor_model
    # dict_state["poisoned_trainset"]=poisoned_trainset
    dict_state["poisoned_ids"]=poisoned_ids
    # dict_state["clean_testset"]=clean_testset
    # dict_state["poisoned_testset"]=poisoned_testset
    dict_state["pattern"] = pattern
    dict_state['weight']=weight
    save_file_name = "dict_state.pth"
    save_path = os.path.join(work_dir, save_file_name)
    torch.save(dict_state, save_path)
    print(f"BadNets攻击完成,数据和日志被存入{save_path}")
    return save_path

def main():
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack_dict_path = attack()
    # 抽取攻击模型和数据并转储
    backdoor_data_save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    create_backdoor_data(attack_dict_path,backdoor_data_save_path)
    # 开始评估
    eval_backdoor(dataset_name,attack_name,model_name)


def update():
    
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", "ImageNet2012_subset", model_name, attack_name, "backdoor_data.pth")
    dict_state_path = os.path.join(exp_root_dir, "ATTACK", "ImageNet2012_subset", model_name, attack_name, "ATTACK_2025-02-21_15:40:27", "dict_state.pth")
    backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
    dict_state = torch.load(dict_state_path,map_location="cpu")

    # 后门模型
    backdoor_model = backdoor_data["backdoor_model"]
    # poisoned_ids
    poisoned_ids = backdoor_data["poisoned_ids"]
    # trigger
    pattern = dict_state["pattern"]
    weight = dict_state['weight']
    # poisoned_trainset
    poisoned_ids_train = poisoned_ids
    poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train, "train")
    poisoned_trainset_fixed = ExtractDataset(poisoned_trainset)
    # poisoned_testset
    poisoned_ids_test = list(range(len(testset)))
    poisoned_testset = gen_poisoned_dataset(poisoned_ids_test, "test")
    poisoned_testset_fixed = ExtractDataset(poisoned_testset)

    new_backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_ids":poisoned_ids,
        "pattern":pattern,
        "weight":weight,
        "poisoned_trainset":poisoned_trainset_fixed,
        "poisoned_testset":poisoned_testset_fixed
    }
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", "ImageNet2012_subset", model_name, attack_name, "backdoor_data.pth")
    torch.save(new_backdoor_data,backdoor_data_path)
    print("save_path:",backdoor_data_path)
    print("update success")



if __name__ == "__main__":
    # main()
    update()

    # proc_title = "Eval|"+dataset_name+"|"+attack_name+"|"+model_name
    # setproctitle.setproctitle(proc_title)
    # print(proc_title)
    # eval_backdoor(dataset_name,attack_name,model_name)
    pass