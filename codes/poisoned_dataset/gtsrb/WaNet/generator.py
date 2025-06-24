

import os
import copy

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torchvision.datasets import DatasetFolder

from codes.core.attacks.WaNet import AddDatasetFolderTrigger, ModifyTarget
from codes.transform_dataset import gtsrb_WaNet
from codes import config
from codes.poisoned_dataset.utils import filter_class


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


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_ids:list,
                 identity_grid, 
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)

        self.poisoned_set = frozenset(poisoned_ids)
        
        # add noise 
        self.noise = noise
        self.noise_set = frozenset([])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(identity_grid, noise_grid,  noise=True))
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        isPoisoned = False
        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
            isPoisoned = True
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            sample = self.poisoned_transform_noise(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # target = self.poisoned_target_transform(target)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target, isPoisoned

def gen_poisoned_dataset(model_name:str,poisoned_ids:list, trainOrtest:str):
    #  数据集
    trainset,testset = gtsrb_WaNet()
    '''
    # 获得训练集transform
    transform_train = Compose([
        ToTensor(), # 在这之前投毒
        RandomHorizontalFlip(),
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    transform_test = Compose([
        ToTensor(), # 在这之前投毒
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    '''
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(
            config.exp_root_dir,
            "ATTACK",
            "GTSRB",
            "ResNet18",
            "WaNet",
            "ATTACK_2024-12-27_13:36:56",
            "dict_state.pth"
        )
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(
            config.exp_root_dir,
            "ATTACK",
            "GTSRB",
            "VGG19",
            "WaNet",
            "ATTACK_2024-12-27_13:37:37",
            "dict_state.pth"
        )
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(
            config.exp_root_dir,
            "ATTACK",
            "GTSRB",
            "DenseNet",
            "WaNet",
            "ATTACK_2024-12-27_13:37:50",
            "dict_state.pth"
        )
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    # trigger
    identity_grid = dict_state["identity_grid"]
    noise_grid = dict_state["noise_grid"]
    # 在最前面进行投毒
    if trainOrtest == "train":
        poisonedDatasetFolder= PoisonedDatasetFolder(trainset,config.target_class_idx, poisoned_ids,identity_grid,noise_grid,noise=False,poisoned_transform_index=0,poisoned_target_transform_index=0)
    elif trainOrtest == "test":
        filtered_testset = filter_class(testset,config.target_class_idx)
        poisonedDatasetFolder= PoisonedDatasetFolder(filtered_testset,config.target_class_idx, poisoned_ids,identity_grid,noise_grid,noise=False,poisoned_transform_index=0,poisoned_target_transform_index=0)
    return poisonedDatasetFolder


