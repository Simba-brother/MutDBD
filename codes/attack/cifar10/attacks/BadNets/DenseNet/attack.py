import os
import time
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor,RandomCrop, RandomHorizontalFlip, ToPILImage

from codes.core.attacks import BadNets

from codes.attack.cifar10.models.densenet import densenet_cifar

import setproctitle

from codes import config
from codes.scripts.dataset_constructor import *
from codes.attack.utils import eval_backdoor,update_backdoor_data
from codes.attack.cifar10.attacks.BadNets.utils import create_backdoor_data

global_seed = config.random_seed
deterministic = True
# cpu种子
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
    np.random.seed(global_seed)
    random.seed(global_seed)

# 训练集transform    
transform_train = Compose([
    # Convert a tensor or an ndarray to PIL Image
    ToPILImage(), 
    RandomCrop(size=32,padding=4,padding_mode="reflect"), 
    RandomHorizontalFlip(), # 随机水平翻转
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ToTensor()
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    ToTensor()
])

# victim model
model = densenet_cifar()
# 获得数据集
trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/train',
    loader=cv2.imread, # ndarray (H,W,C)
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

# backdoor pattern
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
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


exp_root_dir = config.exp_root_dir
dataset_name = "CIFAR10"
model_name = "DenseNet"
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
    'schedule': [100, 150], # epoch区间

    'epochs': 200,

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

    

if __name__ == "__main__":
    # 攻击
    # main()

    # backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    # update_backdoor_data(backdoor_data_path)

    # 评估
    # eval_backdoor(dataset_name,attack_name,model_name)
    pass