'''
24个场景下污染数据集的生成
'''
from codes import config
import copy
import os
import cv2
import torch
from codes.core.attacks.BadNets import AddDatasetFolderTrigger, ModifyTarget
from codes.transform_dataset import imagenet_BadNets
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose
from codes.poisoned_dataset.utils import filter_class

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_ids,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            # 数据集文件夹位置
            benign_dataset.root, # 数据集文件夹 /data/mml/backdoor_detect/dataset/cifar10/train
            # 数据集直接加载器
            benign_dataset.loader, # cv2.imread
            # 数据集扩展名
            benign_dataset.extensions, # .png
            # 数据集transform
            benign_dataset.transform, # 被注入到self.transform
            # 数据集标签transform
            benign_dataset.target_transform, # 对label进行transform
            None)
        # 选出的id set作为污染目标样本
        self.poisoned_set = poisoned_ids
        # Add trigger to images
        # 注意在调用父类（DatasetFolder）构造时self.transform = benign_dataset.transform
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform) # Compose()的深度拷贝
        # 中毒转化器为在普通样本转化器前再加一个AddDatasetFolderTrigger
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))
        # trigger_path = "codes/core/attacks/BadNets_trigger.png"
        # self.poisoned_transform.transforms.insert(poisoned_transform_index, BadNets_transform(trigger_path))
        
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        # DatasetFolder 必须要有迭代
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 父类（DatasetFolder）属性
        sample = self.loader(path) # self.loader也是调用父类构造时注入的
        isPoisoned = False
        if index in self.poisoned_set: # self.poisoned_set 本类构造注入的
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
            isPoisoned = True
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target, isPoisoned

def gen_poisoned_dataset(poisoned_ids:list, trainOrtest:str):
    #  数据集
    trainset,testset = imagenet_BadNets()
    '''
    transform_train = Compose([
        ToPILImage(),
        RandomResizedCrop(size=224), 
        ToTensor()
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    '''

    # backdoor pattern
    pattern = torch.zeros((224, 224), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((224, 224), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    if trainOrtest == "train":
        # 中毒的数据集
        poisonedDatasetFolder = PoisonedDatasetFolder(trainset,config.target_class_idx,poisoned_ids,pattern, weight, -1, 0)
    elif trainOrtest == "test":
        # 中毒的数据集
        # 在数据集转换组合transforms.Compose[]的最后一个元素(ToTensor)之前进行中毒植入
        filtered_testset = filter_class(testset,config.target_class_idx)
        poisonedDatasetFolder = PoisonedDatasetFolder(filtered_testset,config.target_class_idx,poisoned_ids,pattern, weight, -1, 0)
    return poisonedDatasetFolder