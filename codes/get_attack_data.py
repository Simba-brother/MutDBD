import os
import random
import copy
import cv2
import numpy as np
import torch
from torchvision.datasets import DatasetFolder
from codes.core.attacks.IAD import Generator
from codes import config
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop, RandomRotation, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from codes.scripts.dataset_constructor import *


class Add_IAD_DatasetFolderTrigger():
    """Add IAD trigger to DatasetFolder images.
    """

    def __init__(self, modelG, modelM):
         self.modelG = modelG
         self.modelM = modelM

    def __call__(self, img):
        # 允许一个类的实例像函数一样被调用
        """Get the poisoned image..
        img: shap:CHW,type:Tensor
        """
        
        # 添加一个维度索引构成BCHW
        imgs = img.unsqueeze(0) # 增加一个B维度
        # G model生成pattern
        patterns = self.modelG(imgs)
        # 对pattern normalize一下
        patterns = self.modelG.normalize_pattern(patterns)
        # 获得masks
        masks_output = self.modelM.threshold(self.modelM(imgs))
        # inputs, patterns, masks => bd_inputs
        bd_imgs = imgs + (patterns - imgs) * masks_output 
        # 压缩一个维度
        bd_img = bd_imgs.squeeze(0) # 不会replace
        bd_img= bd_img.detach()
        return bd_img

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class IADPoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 modelG,
                 modelM
                 ):
        super(IADPoisonedDatasetFolder, self).__init__(
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
        # 数据集包含的数据量
        total_num = len(benign_dataset)
        # 需要中毒的数据量
        poisoned_num = int(total_num * poisoned_rate)
        # 断言：中毒的数据量必须是>=0
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        # 数据id list
        tmp_list = list(range(total_num)) #[0,1,2,...,N]
        # id list被打乱
        random.shuffle(tmp_list)
        # 选出的id set作为污染目标样本
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        # Add trigger to images
        # 注意在调用父类（DatasetFolder）构造时self.transform = benign_dataset.transform
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform) # Compose()的深度拷贝    
        # 中毒转化器为在普通样本转化器前再加一个AddDatasetFolderTrigger
        self.poisoned_transform.transforms.append(Add_IAD_DatasetFolderTrigger(modelG, modelM))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.append(ModifyTarget(y_target))

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


def get_IAD_dataset(dataset_name):
    if  dataset_name == "CIFAR10":
        # 设置随机种子
        global_seed = 666
        torch.manual_seed(global_seed)

        def _seed_worker(worker_id):
            worker_seed =666
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # 使用BackdoorBox的transform
        transform_train = Compose([
            ToPILImage(),
            Resize((32, 32)),
            RandomCrop((32, 32), padding=5),
            RandomRotation(10),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261))
        ])
        transform_test = Compose([
            ToPILImage(),
            Resize((32, 32)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261)) # imageNet
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
        return trainset,testset,trainset_loader,testset_loader


def get_IAD_backdoor_data():
    dict_state_file_path = os.path.join(
        config.exp_root_dir, 
        "attack", 
        config.dataset_name, 
        config.model_name, 
        "IAD", 
        "attack",
        "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    modelG = Generator("cifar10")
    modelM = Generator("cifar10", out_channels=1)
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])
    modelG.eval()
    modelM.eval()
    trainset,testset,_,_ = get_IAD_dataset(dataset_name=config.dataset_name)

    iad_poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = trainset,
        y_target = 1,
        poisoned_rate = 0.1,
        modelG = modelG,
        modelM =modelM
    )

    iad_poisoned_testset =  IADPoisonedDatasetFolder(
        benign_dataset = testset,
        y_target = 1,
        poisoned_rate = 1,
        modelG = modelG,
        modelM = modelM
    )

    
    backdoor_data = {
        "backdoor_model":dict_state["backdoor_model"],
        "poisoned_trainset":iad_poisoned_trainset,
        "poisoned_testset":iad_poisoned_testset,
        "poisoned_ids" : iad_poisoned_trainset.poisoned_set,
        "clean_testset":testset
    }
    return backdoor_data


                

    

if __name__ == "__main__":
    get_IAD_backdoor_data()





