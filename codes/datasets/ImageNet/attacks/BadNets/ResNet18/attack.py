import os
import time
import cv2
import numpy as np
import random
from collections import defaultdict
import setproctitle

import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize, CenterCrop
from torchvision.models import resnet18

from codes.core.attacks import BadNets
from codes.bigUtils import create_dir
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset
from codes import config


global_seed = 666
deterministic = True
# cpu种子
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
    np.random.seed(global_seed)
    random.seed(global_seed)

# 训练集transform    
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(), # CHW
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ])
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ]),
])

# ImageNet dataset dir
dataset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"

# victim model
model = resnet18(pretrained = True)

# 修改最后一个全连接层的输出类别数量
num_classes = 30  # 假设我们要改变分类数量为30
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, num_classes)

# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(dataset_dir, "train"),
    loader=cv2.imread, # ndarray (H,W,C)
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(dataset_dir, "val"),
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
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)
    
# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
exp_root_dir = config.exp_root_dir 
dataset_name = "ImageNet"
model_name = "ResNet18"
attack_name = "BadNets"
schedule = {
    'device': 'cuda:0',
    
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180], # epoch区间

    'epochs': 200, # 200 attack;10 benign

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack' # attack
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
    # pure clean trainset
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    # pure poisoned trainset
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)

    dict_state = {}
    dict_state["backdoor_model"] = backdoor_model
    dict_state["poisoned_trainset"]=poisoned_trainset
    dict_state["poisoned_ids"]=poisoned_ids
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    dict_state["clean_testset"]=clean_testset
    dict_state["poisoned_testset"]=poisoned_testset
    dict_state["pattern"] = pattern
    dict_state['weight']=weight
    save_file_name = "dict_state.pth"
    save_path = os.path.join(work_dir, save_file_name)
    torch.save(dict_state, save_path)
    print(f"BadNets攻击完成,数据和日志被存入{save_path}")

def eval(model,testset):
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    batch_size = 128
    # 加载trigger set
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    # 测试集总数
    total_num = len(testset_loader.dataset)
    # 评估开始时间
    start = time.time()
    acc = torch.tensor(0., device=device)
    correct_num = 0 # 攻击成功数量
    with torch.no_grad():
        for batch_id, batch in enumerate(testset_loader):
            X = batch[0]
            Y = batch[1]
            X = X.to(device)
            Y = Y.to(device)
            pridict_digits = model(X)
            correct_num += (torch.argmax(pridict_digits, dim=1) == Y).sum()
        acc = correct_num / total_num
        acc = round(acc.item(),3)
    end = time.time()
    print("acc:",acc)
    print(f'Total eval() time: {end-start:.1f} seconds')
    return acc


def process_eval():
    dict_state_file_path = os.path.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")

    backdoor_model = dict_state["backdoor_model"]

    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    

    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    benign_testset_acc = eval(backdoor_model,clean_testset)
    pure_poisoned_trainset_acc = eval(backdoor_model, purePoisonedTrainDataset)
    pure_clean_trainset_acc = eval(backdoor_model, pureCleanTrainDataset)

    print("poisoned_trainset_acc", poisoned_trainset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("clean_testset_acc", benign_testset_acc)
    print("pure_poisoned_trainset_acc", pure_poisoned_trainset_acc)
    print("pure_clean_trainset_acc", pure_clean_trainset_acc)
    print("process_eval() success")

def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")
    dict_state["poisoned_trainset"] = ExtractDataset(dict_state["poisoned_trainset"]) 
    dict_state["poisoned_testset"] = ExtractDataset(dict_state["poisoned_testset"]) 
    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() success")


if __name__ == "__main__":

    setproctitle.setproctitle(dataset_name+"_"+model_name+"_"+attack_name+"_"+"eval")
    # attack()
    # update_dict_state()
    process_eval()
    # get_dict_state()
    pass