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

from codes.core.attacks import Blended

from codes.datasets.MNIST.models.model_1 import CNN_Model_1
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes.utils import create_dir
from collections import defaultdict
from tqdm import tqdm
import setproctitle
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset

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
# 测试集变换
transform_test = Compose([
    ToTensor()
])
# 下载并加载测试集
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)
# victim model


# backdoor pattern
pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 28, 28), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2

blended = Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.1,
    seed=global_seed,
    deterministic=deterministic
)

exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "MNIST"
model_name = "CNN_Model_1"
attack_name = "Blended"

schedule = {
    'device': 'cuda:0',

    'benign_training': False,
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

    'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack'
}

def attack():
    print("Blended开始攻击")
    blended.train(schedule)
    backdoor_model = blended.best_model
    work_dir = blended.work_dir
    poisoned_train_dataset, poisoned_test_dataset = blended.get_poisoned_dataset()
    poisoned_ids = poisoned_train_dataset.poisoned_set
    clean_testset = testset
    poisoned_testset = poisoned_test_dataset
    poisoned_trainset = poisoned_train_dataset
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)
    print("Blended攻击结束,开始保存攻击数据")
    dict_state = {}
    dict_state["backdoor_model"] = backdoor_model
    dict_state["clean_testset"] = clean_testset
    dict_state["poisoned_testset"] = poisoned_testset
    dict_state["poisoned_trainset"] = poisoned_train_dataset
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    torch.save(dict_state, os.path.join(work_dir, "dict_state.pth"))
    print(f"攻击数据被保存到:{os.path.join(work_dir, 'dict_state.pth')}")
    print("attack() finished")



def eval(model,testset):
    '''
    评估接口
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
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name,"attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    # backdoor_model
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model,poisoned_testset)
    clean_testset_acc = eval(backdoor_model,clean_testset)
    purePoisonedTrainDataset_acc = eval(backdoor_model,purePoisonedTrainDataset)
    pureCleanTrainDataset_acc = eval(backdoor_model,pureCleanTrainDataset)

    print("poisoned_trainset_acc", poisoned_trainset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("clean_testset_acc",clean_testset_acc)
    print("purePoisonedTrainDataset_acc",purePoisonedTrainDataset_acc)
    print("pureCleanTrainDataset_acc",pureCleanTrainDataset_acc)


def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name,"attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name,"attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    dict_state["poisoned_trainset"] = ExtractDataset(dict_state["poisoned_trainset"]) 
    dict_state["poisoned_testset"] = ExtractDataset(dict_state["poisoned_testset"]) 
    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() success")


if __name__ == "__main__":
    setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval")
    # update_dict_state()
    process_eval()
    # attack()
    # get_dict_state()
    # update_dict_state()
    pass