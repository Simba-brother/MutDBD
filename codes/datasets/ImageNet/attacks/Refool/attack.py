import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import setproctitle

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize, CenterCrop
from torchvision.datasets import DatasetFolder

# 导入攻击模块
from codes.core import Refool
# 导入模型
from torchvision.models import resnet18,vgg19,densenet121
from codes import config
from codes.datasets.utils import eval_backdoor,update_backdoor_data
from codes.datasets.ImageNet.attacks.Refool.utils import create_backdoor_data

global_seed = config.random_seed
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

exp_root_dir = config.exp_root_dir
dataset_name = "ImageNet2012_subset"
model_name = "ResNet18"
attack_name = "Refool"


num_classes = 30
if model_name == "ResNet18":
    model = resnet18(pretrained = True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
elif model_name == "VGG19":
    deterministic = False
    model = vgg19(pretrained = True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
elif model_name == "DenseNet":
    model = densenet121(pretrained = True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)


# 存储反射照片
reflection_images = []
# URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
# "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" please replace this with path to your desired reflection set
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
# opencv读出来前200个reflection img
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
# 训练集transform
'''
第一版
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor() # CHW
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor()
])
'''

transform_train = Compose([
    ToPILImage(),
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])
transform_test = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])
# 获得数据集
trainset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir,"train"),
    loader=cv2.imread, # ndarray
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root=os.path.join(config.ImageNet2012_subset_dir,"test"),
    loader=cv2.imread,
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

# Configure the attack scheme
refool= Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    poisoned_transform_train_index= 0,
    poisoned_transform_test_index= 0,
    poisoned_target_transform_index= 0,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
    reflection_candidates = reflection_images,
    max_image_size=560,
    ghost_rate=0.49, # default:0.49 焦点模糊比例+ghost比例 = 1.焦点模糊和ghost是refool攻击的2种模式
    alpha_b=0.1, # 默认值：-1。原图占比，直观上讲该值越小攻击效果越好
    offset=(0, 0),
    sigma=5, # default:-1 # 焦点模糊模式中的反射图像的高斯模糊操作，值域为[1,5]，直观上讲越大攻击效果越好。
    ghost_alpha=-1 # default:-1 两种偏移比重
)

schedule = {
    'device': f'cuda:{config.gpu_id}',
    'GPU_num': 4,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.01,
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
    poisoned_testset = refool.poisoned_test_dataset
    poisoned_ids = poisoned_trainset.poisoned_set

    dict_state = {}
    dict_state["poisoned_trainset"]=poisoned_trainset
    dict_state["poisoned_ids"]=poisoned_ids
    dict_state["clean_testset"]=testset
    dict_state["poisoned_testset"]=poisoned_testset

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


def main():
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack_dict_path = attack()
    # 抽取攻击模型和数据并转储
    backdoor_data_save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    create_backdoor_data(attack_dict_path,backdoor_data_save_path)
    # 开始评估
    eval_backdoor(dataset_name,attack_name,model_name)

if __name__ == "__main__":
    main()

    # backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    # update_backdoor_data(backdoor_data_path)

    # eval_backdoor(dataset_name,attack_name,model_name)
    pass