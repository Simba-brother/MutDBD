import sys
sys.path.append("./")
import os
import joblib
import time
import cv2
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

from core.attacks import IAD

from datasets.MNIST.models.model_1 import CNN_Model_1

from codes.common.eval_model import EvalModel
from utils import create_dir
from collections import defaultdict
from tqdm import tqdm
import setproctitle
from codes.scripts.dataset_constructor import IAD_Dataset

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 数据集下载和加载路径
datasets_root_dir = '/data/mml/backdoor_detect/dataset'
dataset = torchvision.datasets.MNIST
victim_model = CNN_Model_1(class_num=10)
# 训练集变换
transform_train = Compose([
    ToTensor()
])
# 下载并加载训练集
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=False)
trainset_1 = dataset(datasets_root_dir, train=True, transform=transform_train, download=False)
# 测试集变换
transform_test = Compose([
    ToTensor()
])
# 下载并加载测试集
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)
testset_1 = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)


exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "MNIST"
model_name = "CNN_Model_1"
attack_name = "IAD"

schedule = {
    'device': 'cuda:0',

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

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
    dataset_name="mnist",
    train_dataset=trainset,
    test_dataset=testset,
    train_dataset1=trainset_1,
    test_dataset1=testset_1,
    model=victim_model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,      # it may not reach the best if we follow the default configure in the original paper
    cross_rate=0.1,         # because we set the target label to 1
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

def eval(model,testset):
    '''
    评估接口
    '''
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
    


def update_dict_state():
    
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    # 加载
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_weight = dict_state["model"]
    # backdoor_model
    victim_model.load_state_dict(backdoor_weight)

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


    dict_state["backdoor_model"] = victim_model
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


if __name__ == "__main__":
    setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval")
    # attack()
    # get_dict_state()
    # update_dict_state()
    process_eval()
    pass