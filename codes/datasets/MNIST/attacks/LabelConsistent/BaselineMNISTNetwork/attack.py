'''
This is the test code of poisoned training under LabelConsistent.
'''

import sys
sys.path.append("./")
import os
import os.path as osp
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from core import LabelConsistent
import setproctitle
from core.models.baseline_MNIST_network import BaselineMNISTNetwork
from codes.scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset

def _seed_worker(worker_id):
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "MNIST"
model_name = "BaselineMNISTNetwork"
attack_name = "LabelConsistent"
global_seed = 666
deterministic = True

torch.manual_seed(global_seed) # cpu随机数种子
victim_model = BaselineMNISTNetwork()
adv_model = BaselineMNISTNetwork()
# 这个是先通过benign训练得到的clean model weight
clean_adv_model_weight_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "benign_attack", "best_model.pth")
adv_model_weight = torch.load(clean_adv_model_weight_path, map_location="cpu")
adv_model.load_state_dict(adv_model_weight)
# 对抗样本保存目录
# 获得数据集

dataset = torchvision.datasets.MNIST
datasets_root_dir = '/data/mml/backdoor_detect/dataset'
transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=False)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)

# 图片四角白点
pattern = torch.zeros((28, 28), dtype=torch.uint8)
k = 6
pattern[:k,:k] = 255
pattern[:k,-k:] = 255
pattern[-k:,:k] = 255
pattern[-k:,-k:] = 255

weight = torch.zeros((28, 28), dtype=torch.float32)
weight[:k,:k] = 1.0
weight[:k,-k:] = 1.0
weight[-k:,:k] = 1.0
weight[-k:,-k:] = 1.0

schedule = {
    'device': 'cuda:0',

    'benign_training': False, # Train Attacked Model
    'batch_size': 1024,
    'num_workers': 8,

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


eps = 64
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.1

label_consistent = LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=BaselineMNISTNetwork(),
    adv_model=adv_model,
    adv_dataset_dir= os.path.join(exp_root_dir,"attack", dataset_name, model_name, attack_name, "adv_dataset", f"eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}"),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

def benign_attack():
    label_consistent.train()


def attack():
    print("LabelConsistent开始攻击")
    label_consistent.train()
    backdoor_model = label_consistent.best_model
    workdir =label_consistent.work_dir
    print("LabelConsistent攻击结束,开始保存攻击数据")
    dict_state = {}
    clean_testset = testset
    poisoned_testset = label_consistent.poisoned_test_dataset
    poisoned_trainset = label_consistent.poisoned_train_dataset
    poisoned_ids = poisoned_trainset.poisoned_set
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)
    dict_state["clean_testset"] = clean_testset
    dict_state["poisoned_testset"] = poisoned_testset
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    dict_state["backdoor_model"] = backdoor_model
    dict_state["poisoned_trainset"] = poisoned_trainset
    dict_state["poisoned_ids"] = poisoned_ids
    torch.save(dict_state, os.path.join(workdir, "dict_state.pth"))
    print(f"攻击数据被保存到:{os.path.join(workdir, 'dict_state.pth')}")
    print("attack() finished")

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
    dict_state_file_path = os.path.join(exp_root_dir, "attack",dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    # backdoor_model
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    
    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model,poisoned_testset)
    clean_testset_acc = eval(backdoor_model,clean_testset)
    purePoisonedTrainDataset_acc = eval(backdoor_model,purePoisonedTrainDataset)
    pureCleanTrainDataset_acc = eval(backdoor_model,pureCleanTrainDataset)
    
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("clean_testset_acc",clean_testset_acc)
    print("purePoisonedTrainDataset_acc",purePoisonedTrainDataset_acc)
    print("pureCleanTrainDataset_acc",pureCleanTrainDataset_acc)
    
    

def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack",dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack",dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    poisoned_trainset=  ExtractDataset(dict_state["poisoned_trainset"])
    dict_state["poisoned_trainset"] = poisoned_trainset
    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() successful")

if __name__ == "__main__":
    setproctitle.setproctitle(attack_name+"_"+model_name+"_attack")
    # benign_attack()
    # attack()
    # process_eval()
    # get_dict_state()
    # update_dict_state()
    pass



