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
import setproctitle
from datasets.GTSRB.models.vgg import VGG
from scripts.dataset_constructor import PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractDataset

def _seed_worker(worker_id):
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)
exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "GTSRB"
model_name = "VGG19"
attack_name = "LabelConsistent"
global_seed = 666
deterministic = True

torch.manual_seed(global_seed) # cpu随机数种子
victim_model = VGG("VGG19", 43)
adv_model = VGG("VGG19", 43)
# 这个是先通过benign训练得到的clean model weight
clean_adv_model_weight_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "benign_attack", "best_model.pth")
adv_model_weight = torch.load(clean_adv_model_weight_path, map_location="cpu")
adv_model.load_state_dict(adv_model_weight)
# 对抗样本保存目录
# 获得数据集
transform_train = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    RandomHorizontalFlip(),
    ToTensor()
])
transform_test = Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/GTSRB/Train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/GTSRB/testset',
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



schedule = {
    'device': 'cuda:1',

    'benign_training': False, # Train Attacked Model
    'batch_size': 256,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 50,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': osp.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack'
}



eps = 16
alpha = 1.5
steps = 100
max_pixel = 255
poisoned_rate = 0.1
label_consistent = core.LabelConsistent(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    adv_model=adv_model,
    adv_dataset_dir= os.path.join(exp_root_dir,"attack", dataset_name, model_name, attack_name, "adv_dataset", f"eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}"),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=poisoned_rate,
    adv_transform=Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), ToTensor()]),
    pattern=pattern,
    weight=weight,
    eps=eps,
    alpha=alpha,
    steps=steps,
    max_pixel=max_pixel,
    poisoned_transform_train_index=2,
    poisoned_transform_test_index=2,
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
    dict_state["poisoned_trainset"] = ExtractDataset(dict_state["poisoned_trainset"])
    dict_state["poisoned_testset"] = ExtractDataset(dict_state["poisoned_testset"])
    torch.save(dict_state, dict_state_file_path)
    print("update_dict_state() successful")

if __name__ == "__main__":
    setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval")
    # attack()
    # benign_attack()
    process_eval()
    # get_dict_state()
    # update_dict_state()
    pass



