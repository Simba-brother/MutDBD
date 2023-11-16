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
from codes import core

from codes.core.models.resnet import ResNet


# CUDA_VISIBLE_DEVICES = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
victim_model = ResNet(18,num_classes=10)
adv_model = ResNet(18,num_classes=10)
# 这个是先通过benign训练得到的clean model weight
adv_model_weight = torch.load('./experiments/cifar10_resnet_nopretrained_32_32_3_labelconsistent_clean_2023-11-14_14:36:52/best_model.pth', map_location="cpu")
adv_model.load_state_dict(adv_model_weight)
# 获得数据集
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])

transform_test = Compose([
    ToTensor()
])
trainset = DatasetFolder(
    root='./dataset/cifar10/train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='./dataset/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)




pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-1, -1] = 255
pattern[-1, -3] = 255
pattern[-3, -1] = 255
pattern[-2, -2] = 255

pattern[0, -1] = 255
pattern[1, -2] = 255
pattern[2, -3] = 255
pattern[2, -1] = 255

pattern[0, 0] = 255
pattern[1, 1] = 255
pattern[2, 2] = 255
pattern[2, 0] = 255

pattern[-1, 0] = 255
pattern[-1, 2] = 255
pattern[-2, 1] = 255
pattern[-3, 0] = 255

weight = torch.zeros((32, 32), dtype=torch.float32)
weight[:3,:3] = 1.0
weight[:3,-3:] = 1.0
weight[-3:,:3] = 1.0
weight[-3:,-3:] = 1.0

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

schedule = {
    'device': 'cuda:1',

    'benign_training': False, # 先训练处来一benign model
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

    'save_dir': 'experiments',
    'experiment_name': 'cifar10_resnet_nopretrained_32_32_3_labelconsistent'
}


eps = 8
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.1

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    adv_model=adv_model,
    # The directory to save adversarial dataset
    adv_dataset_dir=f'experiments/adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
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

def temp():
    dict_state = torch.load("experiments/cifar10_resnet_nopretrained_32_32_3_labelconsistent_2023-11-15_19:52:15/dict_state.pth", map_location="cpu")
    poisoned_trainset = label_consistent.poisoned_train_dataset
    poisoned_ids = poisoned_trainset.poisoned_set
    dict_state["poisoned_trainset"] = poisoned_trainset
    dict_state["poisoned_ids"] = poisoned_ids
    torch.save(dict_state, "experiments/cifar10_resnet_nopretrained_32_32_3_labelconsistent_2023-11-15_19:52:15/dict_state.pth")

def eval(model,testset):
    '''
    评估接口
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
    dict_state = torch.load("experiments/cifar10_resnet_nopretrained_32_32_3_labelconsistent_2023-11-15_19:52:15/dict_state.pth")
    # backdoor_model
    backdoor_model = dict_state["backdoor_model"]
    clean_testset = dict_state["clean_testset"]
    poisoned_testset = dict_state["poisoned_testset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    clean_testset_acc = eval(backdoor_model,clean_testset)
    poisoned_testset_acc = eval(backdoor_model,poisoned_testset)
    pureCleanTrainDataset_acc = eval(backdoor_model,pureCleanTrainDataset)
    purePoisonedTrainDataset_acc = eval(backdoor_model,purePoisonedTrainDataset)
    print("clean_testset_acc",clean_testset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("pureCleanTrainDataset_acc",pureCleanTrainDataset_acc)
    print("purePoisonedTrainDataset_acc",purePoisonedTrainDataset_acc)
    

def get_dict_state():
    dict_state = torch.load("experiments/cifar10_resnet_nopretrained_32_32_3_labelconsistent_2023-11-15_19:52:15/dict_state.pth", map_location="cpu")
    return dict_state

if __name__ == "__main__":
    # attack()
    # process_eval()
    # temp()
    pass



