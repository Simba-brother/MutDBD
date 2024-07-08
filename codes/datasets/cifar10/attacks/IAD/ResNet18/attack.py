import os
import random

import time
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop, RandomRotation, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from codes.core.attacks import IAD
# 生成Backdoor samples
from codes.core.attacks.IAD import Generator 
from codes.core.models.resnet import ResNet
from codes.scripts.dataset_constructor import *
import setproctitle


# 设置随机种子
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 获得一个朴素的resnet18
model = ResNet(num=18,num_classes=10)

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
# 另外一份训练集
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
# 另外一份测试集
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

exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "IAD"
schedule = {
    'device': 'cuda:1',
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

    'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack'
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



def eval(model,testset):
    '''
    model:(ResNet(18)) input shape:(1,32,32,3)
    '''
    model.eval()
    device = torch.device("cuda:1")
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
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    pure_poisoned_trainset = dict_state["purePoisonedTrainDataset"]
    pure_clean_trainset = dict_state["pureCleanTrainDataset"]
    
    assert len(pure_poisoned_trainset)*2 + len(pure_clean_trainset) == len(poisoned_trainset), "数量不对"
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    clean_testset_acc = eval(backdoor_model, clean_testset)
    pure_poisoned_trainset_acc = eval(backdoor_model, pure_poisoned_trainset)
    pure_clean_trainset_acc = eval(backdoor_model, pure_clean_trainset)
    
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("clean_testset_acc",clean_testset_acc)
    print("pure_poisoned_trainset_acc",pure_poisoned_trainset_acc)
    print("pure_clean_trainset_acc",pure_clean_trainset_acc)
    print("process_eval success")
    


def update_dict_state():
    
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    # 加载
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_weight = dict_state["model"]
    # backdoor_model
    model.load_state_dict(backdoor_weight)

    # 污染训练集
    poisoned_trainset_data = dict_state["poisoned_trainset_data"]
    poisoned_trainset_label = dict_state["poisoned_trainset_label"]
    poisoned_trainset = IAD_Dataset(poisoned_trainset_data, poisoned_trainset_label)

    # 污染测试集
    test_poisoned_data = dict_state["test_poisoned_data"]
    test_poisoned_label = dict_state["test_poisoned_label"]
    poisoned_testset = IAD_Dataset(test_poisoned_data, test_poisoned_label)

    # 干净测试集
    clean_testset = testset

    # 纯污染训练集
    pure_poisoned_trainset_data = dict_state["pure_poisoned_trainset_data"]
    pure_poisoned_trainset_label = dict_state["pure_poisoned_trainset_label"]
    pure_poisoned_trainset = IAD_Dataset(pure_poisoned_trainset_data, pure_poisoned_trainset_label)

    # 纯干净训练集
    pure_clean_trainset_data = dict_state["pure_clean_trainset_data"]
    pure_clean_trainset_label = dict_state["pure_clean_trainset_label"]
    pure_clean_trainset = IAD_Dataset(pure_clean_trainset_data, pure_clean_trainset_label)


    dict_state["backdoor_model"] = model
    dict_state["poisoned_trainset"] = poisoned_trainset
    dict_state["poisoned_testset"] = poisoned_testset
    dict_state["clean_testset"] = clean_testset
    dict_state["purePoisonedTrainDataset"] = pure_poisoned_trainset
    dict_state["pureCleanTrainDataset"] = pure_clean_trainset

    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() success")
    


def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    return dict_state
    


def create_backdoor_data():
    # create
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_model = backdoor_model = dict_state["backdoor_model"]

    modelG = Generator("cifar10")
    modelM = Generator("cifar10", out_channels=1)
    
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])

    backdoor_model.eval()
    modelG.eval()
    modelM.eval()

    poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = trainset,
        y_target = 1,
        poisoned_rate = 0.1,
        modelG = modelG,
        modelM =modelM
    )

    poisoned_testset =  IADPoisonedDatasetFolder(
        benign_dataset = testset,
        y_target = 1,
        poisoned_rate = 1,
        modelG = modelG,
        modelM = modelM
    )
    
    # eval
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    clean_testset_acc = eval(backdoor_model, testset)
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("clean_testset_acc",clean_testset_acc)
    # save
    save_dir = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name)
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


def eval_backdoor():
    backdoor_data_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    clean_testset = backdoor_data["clean_testset"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    # eval
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    clean_testset_acc = eval(backdoor_model,clean_testset)
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("clean_testset_acc", clean_testset_acc)


if __name__ == "__main__":
    proc_title = "EvalBackdoor|"+dataset_name+"|"+model_name+"|"+attack_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # attack()
    # create_backdoor_data()
    eval_backdoor()
    pass

