
import os
import cv2
from torchvision.datasets import DatasetFolder
from commonUtils import read_yaml
from datasets.transform import get_cifar10_transform,get_gtsrb_transform,get_imagenet_transform
config = read_yaml("config.yaml")

def get_clean_dataset(dataset_name,attack_name):
    ''' 获得数据集 '''
    if dataset_name == "CIFAR10":
        dataset_dir = config[f"{dataset_name}_dataset_dir"]
        extensions = ('png',)
        train_transform, test_transform = get_cifar10_transform(attack_name)
    elif dataset_name == 'GTSRB':
        dataset_dir = config[f"{dataset_name}_dataset_dir"]
        extensions = ('png',)
        train_transform, test_transform = get_gtsrb_transform(attack_name)
    elif dataset_name == "ImageNet":
        dataset_dir = config["ImageNet2012_subset_dir"]
        extensions = ('jpeg',)
        train_transform, test_transform = get_imagenet_transform(attack_name)

    trainset = DatasetFolder(
        root= os.path.join(dataset_dir, "train"), # 文件夹目录
        loader=cv2.imread, # ndarray
        extensions=extensions,
        transform=train_transform,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root= os.path.join(dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread,
        extensions=extensions,
        transform=test_transform,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset
