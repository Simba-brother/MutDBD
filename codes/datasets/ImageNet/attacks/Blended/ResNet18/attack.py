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
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize, CenterCrop
from torchvision.models import resnet18
from core import Blended
import setproctitle
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset
from codes import config

def _seed_worker(worker_id):
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

global_seed = 666
deterministic = True
torch.manual_seed(global_seed) # 设置 CPU 生成随机数的 种子 ，方便下次复现实验结果。


# 训练集transform    
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(), # CHW
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ])
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ]),
])

dataset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"


# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(dataset_dir, "train"),
    loader=cv2.imread, # ndarray (H,W,C)
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(dataset_dir, "val"),
    loader=cv2.imread, # ndarray(shape:HWC)
    extensions=('jpeg',),
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

pattern = torch.zeros((1, 224, 224), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 224, 224), dtype=torch.float32) # torch.Tensor
weight[0, -3:, -3:] = 0.2

'''
MINIST
pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 28, 28), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2
'''

# victim model
model = resnet18(pretrained = True)
# 冻结预训练模型中所有参数的梯度
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后一个全连接层的输出类别数量
num_classes = 30  # 假设我们要改变分类数量为30
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, num_classes)

blended = Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    pattern=pattern,
    weight=weight,
    y_target=1,
    poisoned_rate=0.1,
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


# Show an Example of Poisoned Testing Samples
# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

exp_root_dir = config.exp_root_dir 
dataset_name = "ImageNet"
model_name = "ResNet18"
attack_name = "Blended"
# Train Benign Model
schedule = {
    'device': 'cuda:0',
    # 'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200, # attack:200|benign:10

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack' # attack|benign
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
    print("process_eval() success")

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
    setproctitle.setproctitle(dataset_name+"_"+model_name+"_"+attack_name+"_"+"eval")
    # attack()
    process_eval()
    # update_dict_state()
    # get_dict_state()
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