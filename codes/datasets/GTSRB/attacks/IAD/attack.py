
import os
import random

import setproctitle
import numpy as np
import cv2
import torch
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor,ToPILImage, Resize, RandomCrop, RandomRotation
from torchvision.datasets import DatasetFolder
# 导入攻击模块
from codes.core.attacks import IAD
# 导入模型
from codes.core.models.resnet import ResNet
from codes.datasets.GTSRB.models.vgg import VGG
from codes.datasets.GTSRB.models.densenet import DenseNet121

from codes import config
from codes.datasets.utils import eval_backdoor,update_backdoor_data
from codes.datasets.GTSRB.attacks.IAD.utils import create_backdoor_data


# 设置随机种子
global_seed = config.random_seed
deterministic = True
torch.manual_seed(global_seed)

exp_root_dir = config.exp_root_dir
dataset_name = "GTSRB"
model_name = "DenseNet"
attack_name = "IAD"

# victim model
num_classes = 43
if model_name == "ResNet18":
    model = ResNet(num = 18, num_classes=num_classes)
elif model_name == "VGG19":
    model = VGG("VGG19", num_classes)
elif model_name == "DenseNet":
    model = DenseNet121(43)

def _seed_worker(worker_id):
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    RandomCrop((32, 32), padding=5),
    RandomRotation(10),
    ToTensor()
])
transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])
# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(config.GTSRB_dataset_dir,"train"),
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
trainset1 = DatasetFolder(
    root=os.path.join(config.GTSRB_dataset_dir,"train"),
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(config.GTSRB_dataset_dir,"test"),
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# 另外一份测试集
testset1 = DatasetFolder(
    root=os.path.join(config.GTSRB_dataset_dir,"test"),
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

schedule = {
    'device': f'cuda:{config.gpu_id}',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'milestones': [100, 200, 300, 400],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300, 400, 500],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 600,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}

# Configure the attack scheme
iad = IAD(
    dataset_name="gtsrb", # 不要变
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    cross_rate=config.poisoned_rate,
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)


def attack():
    iad.train()
    return os.path.join(iad.work_dir,"dict_state.pth")


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
    main()

    # backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    # update_backdoor_data(backdoor_data_path)

    # eval_backdoor(dataset_name,attack_name,model_name)
    pass

