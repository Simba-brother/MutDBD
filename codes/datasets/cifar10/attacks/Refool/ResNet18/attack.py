import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, ToPILImage, Resize, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from codes import core
import setproctitle
from codes.core.models.resnet import ResNet
from codes.scripts.dataset_constructor import *
from codes import config
from codes.datasets.eval_backdoor import eval_backdoor

global_seed = config.random_seed
deterministic = True
torch.manual_seed(global_seed)
def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# target model
model = ResNet(num=18,num_classes=10)
# 存储反射照片
reflection_images = []
# URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
# "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set
reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 

def read_image(img_path, type=None):
    '''
    读取图片
    '''
    img = cv2.imread(img_path)
    # cv2.imshow('Image', img)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError
# reflection image dir下所有的img path
reflection_image_path = os.listdir(reflection_data_dir)
# 读出来前200个reflection img
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
# 训练集transform
transform_train = Compose([
    ToPILImage(),
    # RandomCrop(size=32,padding=4,padding_mode="reflect"), 
    Resize((32, 32)),
    RandomHorizontalFlip(p=1),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])
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

# Configure the attack scheme
refool= core.Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    poisoned_transform_train_index= 1,
    poisoned_transform_test_index= 1,
    poisoned_target_transform_index= 1,
    schedule=None,
    seed=global_seed, # 666
    deterministic=deterministic, # True
    reflection_candidates = reflection_images, # reflection img list
)
exp_root_dir = config.exp_root_dir
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "Refool"
schedule = {
    'device': 'cuda:1',
    'GPU_num': 4,

    'benign_training': False,
    'batch_size': 32,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100, # batch
    'test_epoch_interval': 10, # epoch
    'save_epoch_interval': 10,  # epoch

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}




def attack():
    print("开始attack train")
    refool.train(schedule)
    print("attack train结束")
    work_dir = refool.work_dir
    poisoned_trainset = refool.poisoned_train_dataset
    poisoned_ids = poisoned_trainset.poisoned_set
    poisoned_testset = refool.poisoned_test_dataset
    
    dict_state = {} 
    dict_state["poisoned_trainset"]=poisoned_trainset
    dict_state["poisoned_ids"]=poisoned_ids
    dict_state["poisoned_testset"]=poisoned_testset
    dict_state["clean_testset"]=testset
    
    # 获得backdoor model weights
    backdoor_weights = torch.load(os.path.join(work_dir, "best_model.pth"), map_location="cpu")
    # backdoor model存入字典数据中
    model.load_state_dict(backdoor_weights)
    dict_state["backdoor_model"] = model
    save_file_name = "dict_state.pth"
    save_path = os.path.join(work_dir, save_file_name)
    print("开始保存攻击后数据")
    torch.save(dict_state, save_path)
    print(f"Refool攻击完成,数据和日志被存入{save_path}")
    return save_path



def create_backdoor_data(attack_dict_path):
    dict_state_file_path = os.path.join(attack_dict_path)
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset, poisoned_testset = refool.get_poisoned_dataset()
    poisoned_ids = poisoned_trainset.poisoned_set

    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)

    save_dir = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name)
    save_file_name = "backdoor_data.pth"
    backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_trainset":poisoned_trainset,
        "poisoned_testset":poisoned_testset,
        "clean_testset":testset,
        "poisoned_ids":poisoned_ids
    }
    save_file_path = os.path.join(save_dir,save_file_name)
    torch.save(backdoor_data,save_file_path)
    print(f"backdoor_data is saved in {save_file_path}")
    return save_file_path


def update_backdoor_data():
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    clean_testset = backdoor_data["clean_testset"]
    poisoned_ids = backdoor_data["poisoned_ids"]

    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)
    backdoor_data["poisoned_trainset"] = poisoned_trainset
    backdoor_data["poisoned_testset"] = poisoned_testset

    # 保存数据
    torch.save(backdoor_data, backdoor_data_path)
    print("update_backdoor_data(),successful.")

def main():
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack_dict_path = attack()
    # 抽取攻击模型和数据并转储
    backdoor_data_path = create_backdoor_data(attack_dict_path)
    # 开始评估
    eval_backdoor(dataset_name,attack_name,model_name)

if __name__ == "__main__":
    # main()
    update_backdoor_data()
    # eval_backdoor(dataset_name,attack_name,model_name)
    pass