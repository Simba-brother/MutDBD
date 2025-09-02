
import os
import cv2
import torch
from datasets.clean_dataset import get_clean_dataset

# 数据投毒
from datasets.poisoned_folder.badnets_folder import PoisonedDatasetFolder as BadNetsPoisonedDatasetFolder
from datasets.poisoned_folder.iad_folder import PoisonedDatasetFolder as IADPoisonedDatasetFolder
from datasets.poisoned_folder.refool_folder import PoisonedDatasetFolder as RefoolPoisonedDatasetFolder
from datasets.poisoned_folder.wanet_folder import PoisonedDatasetFolder as WaNetPoisonedDatasetFolder
# 过滤掉原target class样本
from datasets.filter import filter_dataset
from torch.utils.data import Subset
# 用于获得触发器
from mid_data_loader import (
    get_CIFAR10_IAD_attack_dict_path,
    get_CIFAR10_WaNet_attack_dict_path, 
    get_GTSRB_IAD_attack_dict_path, 
    get_GTSRB_WaNet_attack_dict_path, 
    get_backdoor_data)
from attack.core.attacks.IAD import Generator

def get_BadNets_dataset(dataset_name, poisoned_ids):
    clean_train_dataset, clean_test_dataset = get_clean_dataset(dataset_name,"BadNets")
    if dataset_name in ["CIFAR10","GTSRB"]:
        img_size = (32, 32)
    else:
        img_size = (224, 224)
    # backdoor pattern
    pattern = torch.zeros(img_size, dtype=torch.uint8)
    pattern[-3:, -3:] = 255 # 用于归一化前
    weight = torch.zeros(img_size, dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    target_class = 3
    poisoned_trainset = BadNetsPoisonedDatasetFolder(clean_train_dataset,target_class,poisoned_ids,pattern, weight, -1, 0)
    poisoned_testset = BadNetsPoisonedDatasetFolder(clean_test_dataset,target_class,list(range(len(clean_test_dataset))),pattern, weight, -1, 0)
    filtered_ids = filter_dataset(clean_test_dataset,target_class)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return  poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset


def get_IAD_dataset(dataset_name:str, model_name:str,poisoned_ids):
    clean_train_dataset, clean_test_dataset = get_clean_dataset(dataset_name,"IAD")
    # backdoor pattern
    if dataset_name == "CIFAR10":
        attack_dict_path = get_CIFAR10_IAD_attack_dict_path(model_name)
        modelG = Generator("cifar10")
        modelM = Generator("cifar10", out_channels=1)
        dict_state = torch.load(attack_dict_path, map_location="cpu")
        modelG.load_state_dict(dict_state["modelG"])
        modelM.load_state_dict(dict_state["modelM"])
    elif dataset_name == "GTSRB":
        attack_dict_path = get_GTSRB_IAD_attack_dict_path(model_name)
        modelG = Generator("gtsrb")
        modelM = Generator("gtsrb", out_channels=1)
        dict_state = torch.load(attack_dict_path, map_location="cpu")
        modelG.load_state_dict(dict_state["modelG"])
        modelM.load_state_dict(dict_state["modelM"])
    elif dataset_name == "ImageNet2012_subset":
        backdoor_data = get_backdoor_data(dataset_name,model_name,"IAD")
        modelG = Generator("ImageNet")
        modelM = Generator("ImageNet", out_channels=1)
        backdoor_data = torch.load(backdoor_data, map_location="cpu")
        # # 在数据集转换组合transforms.Compose[]的最后进行中毒植入
        modelG.load_state_dict(backdoor_data["modelG"])
        modelM.load_state_dict(backdoor_data["modelM"])
    else:
        raise ValueError("Invalid input")

    modelG.eval()
    modelM.eval()
    target_class = 3
    # 在数据集转换组合transforms.Compose[]的最后进行中毒植入
    poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = clean_train_dataset,
        y_target = target_class,
        poisoned_ids = poisoned_ids,
        modelG = modelG,
        modelM =modelM
    )
    # 投毒测试集
    filtered_ids = filter_dataset(clean_test_dataset,target_class)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    poisoned_testset =  IADPoisonedDatasetFolder(
        benign_dataset = clean_test_dataset,
        y_target = target_class,
        poisoned_ids = list(range(len(clean_test_dataset))),
        modelG = modelG,
        modelM =modelM
    )
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return  poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset

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


def get_Refool_dataset(dataset_name:str, poisoned_ids:list):
    clean_train_dataset, clean_test_dataset = get_clean_dataset(dataset_name,"Refool")
    # backdoor pattern
    reflection_images = []
    # URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    # "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 

    # reflection image dir下所有的img path
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    # 中毒的数据集
    # 在transforms.Compose([])[:1](ToPIL)之后进行投毒
    target_class = 3

    if dataset_name == "CIFAR10":
        poisoned_transform_index=1
    elif dataset_name in ["GTSRB","ImageNet2012_subset"]:
        poisoned_transform_index=0

    poisoned_trainset = RefoolPoisonedDatasetFolder(
        clean_train_dataset, 
        target_class, 
        poisoned_ids, 
        poisoned_transform_index=poisoned_transform_index, 
        poisoned_target_transform_index=1, 
        reflection_cadidates=reflection_images,
        max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.)

    filtered_ids = filter_dataset(clean_test_dataset,target_class)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    poisoned_testset = RefoolPoisonedDatasetFolder(
        clean_test_dataset, 
        target_class, 
        list(range(len(clean_test_dataset))), 
        poisoned_transform_index=poisoned_transform_index, 
        poisoned_target_transform_index=1, 
        reflection_cadidates=reflection_images,
        max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.)
    
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return  poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset

def get_WaNet_dataset(dataset_name:str, model_name:str,poisoned_ids:list):
    clean_train_dataset, clean_test_dataset = get_clean_dataset(dataset_name,"WaNet")
    if dataset_name == "CIFAR10":
        attack_dict_path = get_CIFAR10_WaNet_attack_dict_path(model_name)
        # trigger
        dict_state = torch.load(attack_dict_path, map_location="cpu")
        identity_grid = dict_state["identity_grid"]
        noise_grid = dict_state["noise_grid"]
    elif dataset_name == "GTSRB":
        attack_dict_path = get_GTSRB_WaNet_attack_dict_path(model_name)
        dict_state = torch.load(attack_dict_path, map_location="cpu")
        identity_grid = dict_state["identity_grid"]
        noise_grid = dict_state["noise_grid"]
    elif dataset_name == "ImageNet2012_subset":
        backdoor_data = get_backdoor_data(dataset_name,model_name,"WaNet")
        # trigger
        identity_grid = backdoor_data["identity_grid"]
        noise_grid = backdoor_data["noise_grid"]
    # 在最前面进行投毒
    target_class = 3
    poisoned_trainset = WaNetPoisonedDatasetFolder(clean_train_dataset,target_class, poisoned_ids,identity_grid,noise_grid,noise=False,poisoned_transform_index=-3,poisoned_target_transform_index=0)
    filtered_ids = filter_dataset(clean_test_dataset,target_class)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    poisoned_testset = WaNetPoisonedDatasetFolder(clean_test_dataset,target_class,list(range(len(clean_test_dataset))),identity_grid,noise_grid,noise=False,poisoned_transform_index=-3,poisoned_target_transform_index=0)

    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset

def get_all_dataset(dataset_name:str, model_name:str, attack_name:str, poisoned_ids):
    if attack_name == "BadNets":
        return get_BadNets_dataset(dataset_name, poisoned_ids)
    elif attack_name == "IAD":
        return get_IAD_dataset(dataset_name, model_name, poisoned_ids)
    elif attack_name == "Refool":
        return get_Refool_dataset(dataset_name, poisoned_ids)
    elif attack_name == "WaNet":
        return get_WaNet_dataset(dataset_name, model_name, poisoned_ids)
    else:
        raise ValueError("Invalid input")