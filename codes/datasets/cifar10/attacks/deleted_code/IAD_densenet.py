
import sys
sys.path.append("./")
from typing import Pattern
import os
import random
import pickle
import joblib
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop, RandomRotation, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST

from core.attacks import IAD
from datasets.cifar10.models.densenet import densenet_cifar

# 设置随机种子
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 获得一个朴素的resnet18
model = densenet_cifar()

# 使用BackdoorBox的transform
transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    RandomCrop((32, 32), padding=5),
    RandomRotation(10),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261)) # imageNet
])
# 获得数据集
trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
trainset1 = DatasetFolder(
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
testset1 = DatasetFolder(
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


'''
trainset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
trainset1 = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)
testset1 = CIFAR10(
    root='/data/gaokuofeng/datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)
'''

schedule = {
    'device': 'cuda:0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,

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

    'save_dir': '/data/mml/backdoor_detect/experiments',
    'experiment_name': 'cifar10_densenet_IAD'
}

iad = IAD(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset1,
    test_dataset1=testset1,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # follow the default configure in the original paper
    cross_rate=0.1,         # follow the default configure in the original paper
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

class PoisonedTrainDataset(Dataset):
    def __init__(self, poisoned_trainset_data, poisoned_trainset_label):
        self.poisoned_trainset_data = poisoned_trainset_data
        self.poisoned_trainset_label = poisoned_trainset_label
        
    def __len__(self):
        return len(self.poisoned_trainset_label)
    
    def __getitem__(self, index):
        x = self.poisoned_trainset_data[index]
        y = self.poisoned_trainset_label[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x,y

class PurePoisonedTrainDataset(Dataset):
    def __init__(self, pure_poisoned_trainset_data, pure_poisoned_trainset_label):
        self.pure_poisoned_trainset_data = pure_poisoned_trainset_data
        self.pure_poisoned_trainset_label = pure_poisoned_trainset_label

    def __len__(self):
        return len(self.pure_poisoned_trainset_data)
    
    def __getitem__(self, index):
        x = self.pure_poisoned_trainset_data[index]
        y = self.pure_poisoned_trainset_label[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x,y
    
class PureCleanTrainDataset(Dataset):
    def __init__(self, pure_clean_trainset_data, pure_clean_trainset_label):
        self.pure_clean_trainset_data = pure_clean_trainset_data
        self.pure_clean_trainset_label = pure_clean_trainset_label

    def __len__(self):
        return len(self.pure_clean_trainset_label)
    
    def __getitem__(self, index):
        x = self.pure_clean_trainset_data[index]
        y = self.pure_clean_trainset_label[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x,y
    
class PoisonedTestSet(Dataset):
    def __init__(self, test_poisoned_data, test_poisoned_label):
        self.test_poisoned_data = test_poisoned_data
        self.test_poisoned_label = test_poisoned_label

    def __len__(self):
        return len(self.test_poisoned_label)
    
    def __getitem__(self, index):
        x = self.test_poisoned_data[index]
        y = self.test_poisoned_label[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x,y

class TargetClassCleanTrainDataset(Dataset):
    def __init__(self, pure_clean_trainset_data, pure_clean_trainset_label, target_class):
        self.pure_clean_trainset_data = pure_clean_trainset_data
        self.pure_clean_trainset_label = pure_clean_trainset_label
        self.target_class = target_class
        self.target_class_clean_trainset = self._get_target_class_clean_trainset()

    def _get_target_class_clean_trainset(self):
        target_class_clean_trainset = []
        for sample, label in zip(self.pure_clean_trainset_data, self.pure_clean_trainset_label):
            if label == self.target_class:
                sample = torch.tensor(sample)
                label = torch.tensor(label)
                target_class_clean_trainset.append((sample,label))
        return target_class_clean_trainset
    
    def __len__(self):
        return len(self.target_class_clean_trainset)
    
    def __getitem__(self, index):
        x,y = self.target_class_clean_trainset[index]
        return x,y



def get_config():
    config = {}
    # backdoor model
    backdoor_model = ResNet(18) # input shape:(1,32,32,3)
    # 加载backdoor权重
    dict_data = torch.load("experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/best_ckpt_epoch_601.pth", map_location="cpu")
    backdoor_state_dict = dict_data["model"]
    backdoor_model.load_state_dict(backdoor_state_dict)
    backdoor_model.eval()

    pure_poisoned_trainset_data =  dict_data["pure_poisoned_trainset_data"]
    pure_poisoned_trainset_label = dict_data["pure_poisoned_trainset_label"]
    purePoisonedTrainDataset = PurePoisonedTrainDataset(pure_poisoned_trainset_data, pure_poisoned_trainset_label)

    pure_clean_trainset_data =  dict_data["pure_clean_trainset_data"]
    pure_clean_trainset_label = dict_data["pure_clean_trainset_label"]
    pureCleanTrainDataset = PureCleanTrainDataset(pure_clean_trainset_data, pure_clean_trainset_label)

    config["backdoor_model"] =  backdoor_model
    config["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    config["pureCleanTrainDataset"] = pureCleanTrainDataset
    config["benign_testset"] = testset
    config["poisoned_testset"] = None
    config["target_label"] = 1
    return config


def eval(model,testset):
    '''
    model:(ResNet(18)) input shape:(1,32,32,3)
    '''
    model.eval()
    device = torch.device("cuda:5")
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
    total_num = len(testset_loader.dataset)
    # 评估开始时间
    start = time.time()
    acc = torch.tensor(0., device=device) # 攻击成功率
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

def attack():
    iad.train()
    # work_dir = iad.work_dir
    # dict_state = torch.load(os.path.join(work_dir, "dict_state.pth"))

def process_eval():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state.pth", map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    clean_testset = dict_state["clean_testset"]
    poisoned_testset = dict_state["poisoned_testset"]

    pure_poisoned_trainset = dict_state["purePoisonedTrainDataset"]
    pure_clean_trainset = dict_state["pureCleanTrainDataset"]
    poisoned_trainset = dict_state["poisoned_trainset"]

    clean_testset_acc = eval(backdoor_model, clean_testset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    pure_clean_trainset_acc = eval(backdoor_model, pure_clean_trainset)
    pure_poisoned_trainset_acc = eval(backdoor_model, pure_poisoned_trainset)
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    print("clean_testset_acc",clean_testset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("pure_clean_trainset_acc",pure_clean_trainset_acc)
    print("pure_poisoned_trainset_acc",pure_poisoned_trainset_acc)
    print("poisoned_trainset_acc",poisoned_trainset_acc)

def temp():
    dict_state = torch.load("experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state.pth", map_location="cpu")
    modelG = iad.modelG 
    modelG.load_state_dict(dict_state["modelG"])
    modelM = iad.modelM
    modelM.load_state_dict(dict_state["modelM"])

    device = torch.device("cuda:5")
    modelG.eval()
    modelM.eval()
    modelG.to(device)
    modelM.to(device)
    test_poisoned_data = []
    test_poisoned_label = []
    for inputs1, targets1 in trainset_loader:
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            # Construct the backdoored samples and calculate the backdoored accuracy
            inputs_bd, targets_bd, _, _ = iad.create_bd(inputs1, targets1, modelG, modelM)
            test_poisoned_data += inputs_bd.detach().cpu().numpy().tolist()
            test_poisoned_label += targets_bd.detach().cpu().numpy().tolist()
    dict_state["test_poisoned_data"] = test_poisoned_data
    dict_state["test_poisoned_label"] = test_poisoned_label
    torch.save(dict_state, "experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state_new.pth")


def get_data():
    '''
    得到攻击后的数据,比如backdoor_model污染集等
    '''
    data = {}
    # 加载
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state_new.pth", map_location="cpu")
    backdoor_weight = dict_state["model"]
    # backdoor_model
    model.load_state_dict(backdoor_weight)
    # 干净测试集
    clean_testset = testset
    # 污染测试集
    test_poisoned_data = dict_state["test_poisoned_data"]
    test_poisoned_label = dict_state["test_poisoned_label"]
    poisoned_testset = PoisonedTestSet(test_poisoned_data, test_poisoned_label)

    poisoned_trainset_data = dict_state["poisoned_trainset_data"]
    poisoned_trainset_label = dict_state["poisoned_trainset_label"]
    poisoned_trainset = PoisonedTrainDataset(poisoned_trainset_data, poisoned_trainset_label)

    pure_poisoned_trainset_data = dict_state["pure_poisoned_trainset_data"]
    pure_poisoned_trainset_label = dict_state["pure_poisoned_trainset_label"]
    pure_poisoned_trainset = PurePoisonedTrainDataset(pure_poisoned_trainset_data, pure_poisoned_trainset_label)

    pure_clean_trainset_data = dict_state["pure_clean_trainset_data"]
    pure_clean_trainset_label = dict_state["pure_clean_trainset_label"]
    pure_clean_trainset = PureCleanTrainDataset(pure_clean_trainset_data, pure_clean_trainset_label)

    data["backdoor_model"] = model
    data["clean_testset"] = clean_testset
    data["poisoned_testset"] = poisoned_testset
    data["pure_poisoned_trainset"] = pure_poisoned_trainset
    data["pure_clean_trainset"] = pure_clean_trainset
    return data


def temp_update_dict_state():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state_new.pth", map_location="cpu")
    backdoor_weight = dict_state["model"]
    # backdoor_model
    model.load_state_dict(backdoor_weight)
    # 干净测试集
    clean_testset = testset
    # 污染测试集
    test_poisoned_data = dict_state["test_poisoned_data"]
    test_poisoned_label = dict_state["test_poisoned_label"]
    poisoned_testset = PoisonedTestSet(test_poisoned_data, test_poisoned_label)

    poisoned_trainset_data = dict_state["poisoned_trainset_data"]
    poisoned_trainset_label = dict_state["poisoned_trainset_label"]
    poisoned_trainset = PoisonedTrainDataset(poisoned_trainset_data, poisoned_trainset_label)

    pure_poisoned_trainset_data = dict_state["pure_poisoned_trainset_data"]
    pure_poisoned_trainset_label = dict_state["pure_poisoned_trainset_label"]
    pure_poisoned_trainset = PurePoisonedTrainDataset(pure_poisoned_trainset_data, pure_poisoned_trainset_label)

    pure_clean_trainset_data = dict_state["pure_clean_trainset_data"]
    pure_clean_trainset_label = dict_state["pure_clean_trainset_label"]
    pure_clean_trainset = PureCleanTrainDataset(pure_clean_trainset_data, pure_clean_trainset_label)
    
    dict_state_new = {}
    dict_state_new["backdoor_model"] = model
    dict_state_new["clean_testset"] = clean_testset
    dict_state_new["poisoned_testset"] = poisoned_testset
    dict_state_new["poisoned_trainset"] = poisoned_trainset
    dict_state_new["purePoisonedTrainDataset"] = pure_poisoned_trainset
    dict_state_new["pureCleanTrainDataset"] = pure_clean_trainset
    torch.save(dict_state_new,"/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state_new_new.pth")
    print("temp_update_dict_state() finished")

def get_dict_state():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_IAD_2023-11-08_23:13:28/dict_state.pth", map_location="cpu")
    return dict_state

if __name__ == "__main__":
    attack()
    # process_eval()
    pass

