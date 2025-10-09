'''
This is the test code of poisoned training under LabelConsistent.
'''

import sys
sys.path.append("./")
import os
import copy
import os.path as osp
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip,Normalize
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from codes import core
from codes import config
import setproctitle
from codes.core.models.resnet import ResNet
from codes.scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset
from codes.common.time_handler import get_formattedDateTime

def _seed_worker(worker_id):
    worker_seed =0
    np.random.seed(worker_seed)
    random.seed(worker_seed)
exp_root_dir = config.exp_root_dir
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "LabelConsistent"
global_seed = 0
deterministic = True
benign_training_flag = False
gpu_id = 0

torch.manual_seed(global_seed) # cpu随机数种子
victim_model = ResNet(18,num_classes=10)
adv_model = None
# 攻击时才打开，良性时注释掉
adv_model = copy.deepcopy(victim_model)
benign_state_dict_path = os.path.join(exp_root_dir,"ATTACK",dataset_name, model_name, attack_name, "benign_train_2025-07-16_13:17:28", "best_model.pth")
benign_state_dict = torch.load(benign_state_dict_path, map_location="cpu")
adv_model.load_state_dict(benign_state_dict)

# 对抗样本保存目录
# 获得数据集
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = DatasetFolder(
    root=os.path.join(config.CIFAR10_dataset_dir,"train"),
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root=os.path.join(config.CIFAR10_dataset_dir,"test"),
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


# 图片四角白点
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



_time = get_formattedDateTime()
schedule = {
    'device': f'cuda:{gpu_id}',

    'benign_training': benign_training_flag,
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

    'save_dir': osp.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': "attack_train"
}


eps = 8 # Maximum perturbation for PGD adversarial attack. Default: 8.
alpha = 1.5 # Step size for PGD adversarial attack. Default: 1.5.
steps = 100 # Number of steps for PGD adversarial attack. Default: 100.
max_pixel = 255
poisoned_rate = 0.1 # 0.1 # 目标类别的0.1

label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    adv_model=adv_model,
    # The directory to save adversarial dataset
    adv_dataset_dir=os.path.join(exp_root_dir,"ATTACK", dataset_name, model_name, attack_name, "adv_dataset", f"eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}"),
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
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
    # workdir =label_consistent.work_dir
    print("LabelConsistent攻击结束,开始保存攻击数据")
    dict_state = {}
    poisoned_testset = label_consistent.poisoned_test_dataset
    # poisoned_trainset = label_consistent.poisoned_train_dataset
    poisoned_ids = label_consistent.poisoned_set
    # clean_testset = testset
    # pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    # purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)
    dict_state["poisoned_ids"] = poisoned_ids
    dict_state["poisoned_testset"] = poisoned_testset
    dict_state["backdoor_model"] = backdoor_model
    
    save_path = os.path.join(
        config.exp_root_dir, "ATTACK",
        dataset_name, model_name, attack_name,
        "backdoor_data.pth")
    torch.save(dict_state, save_path)
    print(f"攻击结果保存到:{save_path}")
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
    
    print("process_eval() success")

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

    
    # setproctitle.setproctitle(dataset_name+"|"+model_name+"|"+attack_name+"|"+"BenignTrain")
    # benign_attack()
    

    setproctitle.setproctitle(dataset_name+"|"+model_name+"|"+attack_name+"|"+"ATTACK")
    attack()

    # process_eval()
    # get_dict_state()
    # update_dict_state()
    pass



