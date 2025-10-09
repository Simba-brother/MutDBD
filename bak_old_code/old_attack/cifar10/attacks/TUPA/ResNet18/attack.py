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
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

from core.attacks import TUAP

from core.models.resnet import ResNet
# from modelMutat import ModelMutat
from codes.common.eval_model import EvalModel
from utils import create_dir
from collections import defaultdict
from tqdm import tqdm
import setproctitle
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset

global_seed = 666
deterministic = True
# cpu种子
torch.manual_seed(global_seed)

def _seed_worker(worker_id):
    np.random.seed(global_seed)
    random.seed(global_seed)

# 训练集transform    
transform_train = Compose([
    # Convert a tensor or an ndarray to PIL Image
    ToPILImage(), 
    # 训练数据增强
    RandomHorizontalFlip(), # 随机水平翻转
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ToTensor()
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    ToTensor()
])

# victim model
model = ResNet(num=18,num_classes=10)
# 获得数据集
trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/train',
    loader=cv2.imread, # ndarray (H,W,C)
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

exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "TUPA"
proctitle = dataset_name+"|"+model_name+"|"+attack_name+"|"+"benign"
setproctitle.setproctitle(proctitle)
print(f"proctitle:{proctitle}")

schedule = {
    'device': 'cuda:0',
    'benign_training': True, # Train Attacked Model
    'batch_size': 128,
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
    'save_dir': osp.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': "benign"
}
poisoned_rate = 0.1
epsilon = 10
# epsilon = 0.031
delta = 0.2
max_iter_uni = np.inf
p_norm = np.inf
num_classes = 10
overshoot = 0.02
max_iter_df = 50
p_samples = 0.01
mask = np.ones((3, 32, 32))

tuap = TUAP(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),

    benign_model=model,
    y_target=1,
    poisoned_rate=poisoned_rate,
    epsilon = epsilon,
    delta=delta,
    max_iter_uni=max_iter_uni,
    p_norm=p_norm,
    num_classes=num_classes,
    overshoot=overshoot,
    max_iter_df=max_iter_df,
    p_samples=p_samples,
    mask=mask,
    poisoned_transform_train_index=-1,
    poisoned_transform_test_index=-1,
    poisoned_target_transform_index=0,
    schedule=schedule,
    seed=global_seed,
    deterministic=True
)

def benign_attack():
    tuap.train()

def attack():
    # 攻击
    tuap.train()
    # 工作dir
    work_dir = tuap.work_dir
    # 获得backdoor model weights
    backdoor_model = tuap.best_model
    # clean testset
    clean_testset = testset
    # poisoned testset
    poisoned_testset = tuap.poisoned_test_dataset
    # poisoned trainset
    poisoned_trainset = tuap.poisoned_train_dataset
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
    save_file_name = "dict_state.pth"
    save_path = osp.join(work_dir, save_file_name)
    torch.save(dict_state, save_path)
    print(f"TUPA攻击完成,数据和日志被存入{save_path}")

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
    dict_state_file_path = osp.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")

    backdoor_model = dict_state["backdoor_model"]

    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    

    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    benign_testset_acc = eval(backdoor_model,clean_testset)
    pure_poisoned_trainset_acc = eval(backdoor_model, purePoisonedTrainDataset)
    pure_clean_trainset_acc = eval(backdoor_model, pureCleanTrainDataset)

    print("poisoned_trainset_acc", poisoned_trainset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("clean_testset_acc", benign_testset_acc)
    print("pure_poisoned_trainset_acc", pure_poisoned_trainset_acc)
    print("pure_clean_trainset_acc", pure_clean_trainset_acc)
    print("process_eval() success")

def get_dict_state():
    dict_state_file_path = osp.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")
    return dict_state

def update_dict_state():
    dict_state_file_path = osp.join(exp_root_dir,"attack",dataset_name,model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path,map_location="cpu")
    dict_state["poisoned_trainset"] = ExtractDataset(dict_state["poisoned_trainset"]) 
    dict_state["poisoned_testset"] = ExtractDataset(dict_state["poisoned_testset"]) 

    # clean_testset = dict_state["clean_testset"]
    # clean_testset = ExtractDataset(dict_state["clean_testset"]) 
    # dict_state["clean_testset"] = clean_testset

    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() success")

def insert_dict_state():
    pass



if __name__ == "__main__":
    benign_attack()
    # attack()
    # update_dict_state()
    # process_eval()
    # get_dict_state()
    pass