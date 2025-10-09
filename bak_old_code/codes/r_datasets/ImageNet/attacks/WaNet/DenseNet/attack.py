'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''
import sys
sys.path.append("./")
import os
import joblib
import random
import time
import setproctitle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize,RandomResizedCrop,CenterCrop
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
from torchvision.models import densenet121

from core import WaNet
from codes import config
from codes.scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset

# if global_seed = 666, the network will crash during training on MNIST. Here, we set global_seed = 555.
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker():
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

# target model
model = densenet121(pretrained = True)
# 冻结预训练模型中所有参数的梯度
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后一个全连接层的输出类别数量
num_classes = 30  # 假设我们要改变分类数量为30
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
# 训练集transform
transform_train = Compose([
    ToPILImage(), 
    # Resize((224, 224)),  # 调整图像大小
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

identity_grid,noise_grid=gen_grid(224,4)

wanet = WaNet(
    train_dataset=trainset, # type:Dataset
    test_dataset=testset, # type:Dataset
    model=model,
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
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
dataset_name = "ImageNet"
model_name = "DenseNet"
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

    'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack'
}


def eval(model, testset):
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
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


def process_eval():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    
    poisoned_trainset_acc = eval(backdoor_model, poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    clean_testset_acc = eval(backdoor_model,clean_testset)
    pure_poisoned_trainset_acc = eval(backdoor_model, purePoisonedTrainDataset)
    pure_clean_trainset_acc = eval(backdoor_model, pureCleanTrainDataset)
    
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("clean_testset_acc", clean_testset_acc)
    print("pure_poisoned_trainset_acc", pure_poisoned_trainset_acc)
    print("pure_clean_trainset_acc", pure_clean_trainset_acc)
    
    print("process_eval success")

def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    dict_state["poisoned_trainset"] = ExtractDataset(dict_state["poisoned_trainset"])
    dict_state["poisoned_testset"] = ExtractDataset(dict_state["poisoned_testset"])
    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state(), success")

if __name__ == "__main__":
    setproctitle.setproctitle(dataset_name+"_"+model_name+"_"+attack_name+"_"+"eval")
    # attack()
    # get_dict_state()
    process_eval()
    # update_dict_state()
    pass
    # attack()
    # infected_model = wanet.get_model()
    
    # # Test Infected Model
    # test_schedule = {
    #     'device': 'GPU',
    #     'CUDA_VISIBLE_DEVICES': '2',
    #     'GPU_num': 1,

    #     'batch_size': 128,
    #     'num_workers': 4,

    #     'save_dir': 'experiments',
    #     'experiment_name': 'test_poisoned_CIFAR10_WaNet'
    # }
    # wanet.test(test_schedule)
