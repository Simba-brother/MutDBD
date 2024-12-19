'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''
import os
import joblib
import random
import time
import setproctitle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from codes.core import WaNet
from codes.datasets.cifar10.models.vgg import VGG
from codes.scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset
from codes import config
from codes.datasets.eval_backdoor import eval_backdoor

# if global_seed = 666, the network will crash during training on MNIST. Here, we set global_seed = 555.
global_seed = config.random_seed
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
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

# model
model = VGG("VGG19")
# transform
# 获得transform
# 获得训练集transform
transform_train = Compose([
    ToTensor(),
    RandomCrop(size=32,padding=4,padding_mode="reflect"),
    RandomHorizontalFlip()
])
# 获得测试集transform
transform_test = Compose([
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
# 获得加载器
batch_size = 128
trainset_loader = DataLoader(
    trainset,
    batch_size = batch_size,
    shuffle=True,
    # num_workers=self.current_schedule['num_workers'],
    drop_last=False,
    pin_memory=False,
    worker_init_fn=_seed_worker
    )
testset_loader = DataLoader(
    testset,
    batch_size = batch_size,
    shuffle=False,
    # num_workers=self.current_schedule['num_workers'],
    drop_last=False,
    pin_memory=False,
    worker_init_fn=_seed_worker
    )

# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])
# trainset = dataset('../datasets', train=True, transform=transform_train, download=False)

# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('../datasets', train=False, transform=transform_test, download=False)


# Show an Example of Benign Training Samples
# index = 44

# x, y = trainset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

identity_grid,noise_grid=gen_grid(32,4)

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
    noise=False,
    poisoned_transform_train_index=-3,
    poisoned_transform_test_index=-3,
    poisoned_target_transform_index=0,
    seed=global_seed,
    deterministic=deterministic
)



# Show an Example of Poisoned Training Samples
# x, y = poisoned_train_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# # Show an Example of Poisoned Testing Samples
# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


exp_root_dir = config.exp_root_dir
dataset_name = "CIFAR10"
model_name = "VGG19"
attack_name = "WaNet"
schedule = {
    'device': 'cuda:1',  # | cpu

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    # 优化器需要的
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180], # 在 150和180epoch时调整lr

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
    # pure clean trainset
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    # pure poisoned trainset
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)

    work_dir = wanet.work_dir
    backdoor_weight = torch.load(os.path.join(work_dir, "best_model.pth"), map_location="cpu")
    model.load_state_dict(backdoor_weight)
    dict_state = {}
    # 中毒训练集
    dict_state["poisoned_trainset"]=poisoned_trainset
    # 中毒样本ids
    dict_state["poisoned_ids"]=poisoned_ids
    # 中毒训练集中纯干净数据集
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    # 中毒训练集中纯中毒数据集
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
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


def create_backdoor_data(attack_dict_path):
    dict_state_file_path = os.path.join(attack_dict_path)
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = wanet.poisoned_train_dataset
    poisoned_testset = wanet.poisoned_test_dataset

     # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)

    save_dir = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name)
    save_file_name = "backdoor_data.pth"
    backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_trainset":poisoned_trainset,
        "poisoned_testset":poisoned_testset,
        "clean_testset":testset,
        "poisoned_ids":poisoned_trainset.poisoned_set
    }
    save_file_path = os.path.join(save_dir,save_file_name)
    torch.save(backdoor_data,save_file_path)
    print(f"backdoor_data is saved in {save_file_path}")


def update_backdoor_data():
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    clean_testset = backdoor_data["clean_testset"]
    poisoned_ids = backdoor_data["poisoned_ids"]

    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)
    backdoor_data["poisoned_trainset"] = poisoned_trainset
    backdoor_data["poisoned_testset"] = poisoned_testset

    # 保存数据
    torch.save(backdoor_data, backdoor_data_path)
    print("update_backdoor_data(),successful.")

def main():
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack_dict_path = attack()
    # 抽取攻击模型和数据并转储
    backdoor_data_path = create_backdoor_data(attack_dict_path)
    # 开始评估
    eval_backdoor(dataset_name,attack_name,model_name)

if __name__ == "__main__":
    # main()
    update_backdoor_data()
    # eval_backdoor(dataset_name,attack_name,model_name)
    pass

