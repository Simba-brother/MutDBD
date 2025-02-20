'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''
import os
import random
import time
import setproctitle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize, CenterCrop
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
# 导入攻击模块
from codes.core import WaNet
# 导入模型
from torchvision.models import resnet18,vgg19,densenet121
# 导入配置
from codes import config
# 导入辅助工具
from codes.datasets.utils import eval_backdoor,update_backdoor_data
from codes.datasets.ImageNet.attacks.WaNet.utils import create_backdoor_data

global_seed = config.random_seed
deterministic = True
torch.manual_seed(global_seed)

exp_root_dir = config.exp_root_dir
dataset_name = "ImageNet2012_subset"
model_name = "ResNet18"
attack_name = "WaNet"

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

def _seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    height = 32
    k = 4
    """
    # shape:(1, 2, k, k), 均匀分布 从区间[0,1)的均匀分布中随机抽取 ndarray
    ins = torch.rand(1, 2, k, k) * 2 - 1 # 区间变为（-1，1）
    # 先去取tensor的绝对值=>均值=>所有数据再去除以均值
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2
    return identity_grid, noise_grid

'''
原来的

# 获得训练集transform
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor(),
])
'''
# 获得训练集transform
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    ToTensor()
])
transform_test = Compose([
    ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    ToTensor()
])
# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir,"train"),
    loader=cv2.imread, # ndarray
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir,"test"),
    loader=cv2.imread,
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
# 获得加载器
batch_size = 128

identity_grid,noise_grid=gen_grid(224,8)

wanet = WaNet(
    train_dataset=trainset, # type:Dataset
    test_dataset=testset, # type:Dataset
    model=model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=deterministic,
    s=0.9
)

schedule = {
    'device': f'cuda:{config.gpu_id}',

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    # 优化器需要的
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 150], # 在 150和180epoch时调整lr

    'epochs': 200,

    'log_iteration_interval': 100, # 每过100个batch,记录下日志
    'test_epoch_interval': 10, # 每经过10个epoch,去测试下model效果
    'save_epoch_interval': 10, # 每经过10个epoch,保存下训练的model ckpt

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}


def attack():
    
    print("wanet后门攻击训练开始")
    wanet.train(schedule)
    print("开始保存攻击后的重要数据")
    # clean testset
    clean_testset = testset
    # poisoned testset
    poisoned_testset = wanet.poisoned_test_dataset

    # poisoned trainset
    poisoned_trainset = wanet.poisoned_train_dataset
    # poisoned_ids
    poisoned_ids = poisoned_trainset.poisoned_set

    work_dir = wanet.work_dir
    backdoor_weight = torch.load(os.path.join(work_dir, "best_model.pth"), map_location="cpu")
    model.load_state_dict(backdoor_weight)
    dict_state = {}
    # 中毒训练集
    dict_state["poisoned_trainset"]=poisoned_trainset
    # 中毒样本ids
    dict_state["poisoned_ids"]=poisoned_ids
    # 干净的测试集
    dict_state["clean_testset"]=testset
    # 中毒的测试集
    dict_state["poisoned_testset"]=poisoned_testset
    dict_state["backdoor_model"] = model
    dict_state["identity_grid"]=identity_grid
    dict_state["noise_grid"]=noise_grid
    save_path = os.path.join(work_dir,"dict_state.pth")
    torch.save(dict_state, save_path)
    print(f"数据被保存在:{save_path}")
    return save_path


def main():
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack_dict_path = attack()
      # 抽取攻击模型和数据并转储
    backdoor_data_save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    create_backdoor_data(attack_dict_path, backdoor_data_save_path)
    # 开始评估
    eval_backdoor(dataset_name, attack_name, model_name)

if __name__ == "__main__":
    main()


    # backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    # update_backdoor_data(backdoor_data_path)
    
    # eval_backdoor(dataset_name,attack_name,model_name)
    pass

