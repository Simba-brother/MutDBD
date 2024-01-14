import sys
sys.path.append("./")
import os.path as osp
import joblib
import time
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

from codes.core.attacks import BadNets

from codes.core.models.resnet import ResNet
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes import draw
from codes.utils import create_dir
from collections import defaultdict
from tqdm import tqdm

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 训练集transform    
transform_train = Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    ToTensor()
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    ToTensor()
])

# target model
model = ResNet(num=18,num_classes=10)
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
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

class PureCleanTrainDataset(Dataset):
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.pureCleanTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        pureCleanTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label = self.poisoned_train_dataset[id]
            if id not in self.poisoned_ids:
                pureCleanTrainDataset.append((sample,label))
        return pureCleanTrainDataset
    
    def __len__(self):
        return len(self.pureCleanTrainDataset)
    
    def __getitem__(self, index):
        x,y=self.pureCleanTrainDataset[index]
        return x,y


class PurePoisonedTrainDataset(Dataset):
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.purePoisonedTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        purePoisonedTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label = self.poisoned_train_dataset[id]
            if id in self.poisoned_ids:
                purePoisonedTrainDataset.append((sample,label))
        return purePoisonedTrainDataset
    
    def __len__(self):
        return len(self.purePoisonedTrainDataset)
    
    def __getitem__(self, index):
        x,y=self.purePoisonedTrainDataset[index]
    
        return x,y
    
class ExtractDataset(Dataset):
    def __init__(self, old_dataset):
        self.old_dataset = old_dataset
        self.new_dataset = self._get_new_dataset()

    def _get_new_dataset(self):
        new_dataset = []
        for id in range(len(self.old_dataset)):
            sample, label = self.old_dataset[id]
            new_dataset.append((sample,label))
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        x,y=self.new_dataset[index]
        return x,y
    
# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'cuda:1',
    
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': '/data/mml/backdoor_detect/experiments',
    'experiment_name': 'cifar10_resnet18_nopretrain_32_32_3_badnets'
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
    save_path = osp.join(work_dir, save_file_name)
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
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth",map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    clean_testset = dict_state["clean_testset"]
    poisoned_testset = dict_state["poisoned_testset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    poisoned_trainset = dict_state["poisoned_trainset"]

    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    benign_testset_acc = eval(backdoor_model,clean_testset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    pure_clean_trainset_acc = eval(backdoor_model, pureCleanTrainDataset)
    pure_poisoned_trainset_acc = eval(backdoor_model, purePoisonedTrainDataset)
    print("poisoned_trainset_acc", poisoned_trainset_acc)
    print("clean_testset_acc", benign_testset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("pure_clean_trainset_acc", pure_clean_trainset_acc)
    print("pure_poisoned_trainset_acc", pure_poisoned_trainset_acc)

def get_dict_state():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth", map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth", map_location="cpu")
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_trainset = ExtractDataset(dict_state["poisoned_trainset"]) 
    dict_state["poisoned_trainset"] = poisoned_trainset
    torch.save(dict_state, "/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth")




if __name__ == "__main__":
    # update_dict_state()
    # attack()
    process_eval()
    # get_dict_state()
    # gf_mutate()
    # neuron_activation_inverse_mutate()
    # neuron_block_mutate()
    # neuron_switch_mutate()
    # weight_shuffling()
    # eval_mutated_model()
    # draw_box()
    pass