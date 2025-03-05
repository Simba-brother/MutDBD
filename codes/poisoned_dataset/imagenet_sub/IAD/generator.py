import os
import copy
import torch
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose

from codes.transform_dataset import imagenet_IAD
from codes import config
from codes.core.attacks.IAD import Generator
from codes.scripts.dataset_constructor import Add_IAD_DatasetFolderTrigger,ModifyTarget

class IADPoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_ids:list,
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
        self.poisoned_set = poisoned_ids
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



def gen_poisoned_dataset(model_name:str,poisoned_ids:list,trainOrtest:str):
    #  数据集
    trainset,trainset1, testset, testset1 = imagenet_IAD()
    '''
    transform_train = Compose([
        ToPILImage(),
        Resize((224, 224)),
        RandomCrop((224, 224), padding=5),
        RandomRotation(10),
        ToTensor()
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    '''
    backdoor_data = os.path.join(
        config.exp_root_dir,
        "ATTACK",
        "ImageNet2012_subset",
        f"{model_name}",
        "IAD",
        "backdoor_data.pth"
    )
    modelG = Generator("ImageNet")
    modelM = Generator("ImageNet", out_channels=1)
    backdoor_data = torch.load(backdoor_data, map_location="cpu")

    # # 在数据集转换组合transforms.Compose[]的最后进行中毒植入
    modelG.load_state_dict(backdoor_data["modelG"])
    modelM.load_state_dict(backdoor_data["modelM"])

    modelG.eval()
    modelM.eval()

    if trainOrtest == "train":
        poisonedDatasetFolder =  IADPoisonedDatasetFolder(
            benign_dataset = trainset,
            y_target = config.target_class_idx,
            poisoned_ids = poisoned_ids,
            modelG = modelG,
            modelM =modelM
        )
    elif trainOrtest == "test":
        poisonedDatasetFolder =  IADPoisonedDatasetFolder(
            benign_dataset = testset,
            y_target = config.target_class_idx,
            poisoned_ids = poisoned_ids,
            modelG = modelG,
            modelM =modelM
        )
    return poisonedDatasetFolder
    
    