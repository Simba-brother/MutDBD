'''
This is the test code of benign training and poisoned training under Blended Attack.
'''

import sys
sys.path.append("./")
import time
import numpy as np
import random
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from codes import core

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR10
# dataset = torchvision.datasets.MNIST



transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
# trainset = dataset('data', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
# testset = dataset('data', train=False, transform=transform_test, download=True)
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

# Show an Example of Benign Training Samples
# index = 44

# x, y = trainset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# Settings of Pattern and Weight

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2

'''
pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 28, 28), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2
'''

victim_model = core.models.ResNet(18,num_classes=10)

blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.1,
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


# Show an Example of Poisoned Training Samples
# x, y = poisoned_train_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# Show an Example of Poisoned Testing Samples
# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# Train Benign Model
schedule = {
    'device': 'cuda:3',
    # 'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

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
    # 'experiment_name': 'train_benign_CIFAR10_Blended'
    'experiment_name': 'cifar10_resnet_nopretrained_32_32_3_Blended'
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
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_Blended_2023-11-15_18:07:35/dict_state_new.pth")
    # backdoor_model
    backdoor_model = dict_state["backdoor_model"]
    clean_testset = dict_state["clean_testset"]
    clean_testset.root = "/data/mml/backdoor_detect/dataset/cifar10/test"
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
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrained_32_32_3_Blended_2023-11-15_18:07:35/dict_state.pth", map_location="cpu")
    return dict_state

if __name__ == "__main__":
    # attack()
    # get_dict_state()
    # process_eval()
    pass

'''
# Test Benign Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    # 'experiment_name': 'test_benign_CIFAR10_Blended'
    'experiment_name': 'test_benign_MNIST_Blended'
}
blended.test(test_schedule)

blended.model = core.models.BaselineMNISTNetwork()
# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

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

    'save_dir': 'experiments',
    # 'experiment_name': 'train_poisoned_CIFAR10_Blended'
    'experiment_name': 'train_poisoned_MNIST_Blended'
}

blended.train(schedule)
infected_model = blended.get_model()


# Test Infected Model
test_schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'batch_size': 128,
    'num_workers': 4,

    'save_dir': 'experiments',
    # 'experiment_name': 'test_poisoned_CIFAR10_Blended'
    'experiment_name': 'test_poisoned_MNIST_Blended'
}
blended.test(test_schedule)
'''