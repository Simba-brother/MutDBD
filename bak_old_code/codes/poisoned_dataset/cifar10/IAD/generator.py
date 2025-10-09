import os
import copy
import torch
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose,Normalize

from codes.transform_dataset import cifar10_IAD
from codes import config
from codes.core.attacks.IAD import Generator
from codes.scripts.dataset_constructor import Add_IAD_DatasetFolderTrigger,ModifyTarget
from codes.poisoned_dataset.utils import filter_class
from torch.utils.data import DataLoader,Subset
# from codes.look_poisoned_img import reverse_normalize, tensor_to_PIL

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




def get_attack_dict_path(model_name:str):
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(config.exp_root_dir,
                                    "ATTACK",
                                    "CIFAR10",
                                    f"{model_name}",
                                    "IAD",
                                    "ATTACK_2024-12-18_13:17:49",
                                    "dict_state.pth")
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(config.exp_root_dir,
                            "ATTACK",
                            "CIFAR10",
                            f"{model_name}",
                            "IAD",
                            "ATTACK_2024-12-18_13:20:48",
                            "dict_state.pth")
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(config.exp_root_dir,
                            "ATTACK",
                            "CIFAR10",
                            f"{model_name}",
                            "IAD",
                            "ATTACK_2024-12-18_13:24:29",
                            "dict_state.pth")

    return attack_dict_path

def gen_needed_dataset(model_name:str,poisoned_ids:list):
    #  数据集
    trainset,trainset1, testset, testset1 = cifar10_IAD()
    '''
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        RandomCrop((32, 32), padding=5),
        RandomRotation(10),
        RandomHorizontalFlip(p=0.5),
        # 图像数据归一化
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        # 在这里append了中毒trans
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        # 归一化
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        # 在这里append了中毒trans
    ])
    '''

    # backdoor pattern
    attack_dict_path = get_attack_dict_path(model_name)
    modelG = Generator("cifar10")
    modelM = Generator("cifar10", out_channels=1)
    
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])

    modelG.eval()
    modelM.eval()

    # 在数据集转换组合transforms.Compose[]的最后进行中毒植入
    poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = trainset,
        y_target = 3,
        poisoned_ids = poisoned_ids,
        modelG = modelG,
        modelM =modelM
    )
    # 投毒测试集
    clean_testset_label_list = []
    clean_testset_loader = DataLoader(
                testset,
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(testset)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != 3:
            filtered_ids.append(sample_id)
    poisoned_testset =  IADPoisonedDatasetFolder(
        benign_dataset = testset,
        y_target = 3,
        poisoned_ids = list(range(len(testset))),
        modelG = modelG,
        modelM =modelM
    )
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return  poisoned_trainset, filtered_poisoned_testset, trainset, testset


if __name__ == "__main__":
    model_name = "ResNet18"
    poisoned_ids = [0,1,2,3,4,5,6,7,8,9,10]
    poisoned_trainset = gen_needed_dataset(model_name,poisoned_ids)
    sample_id = 0
    sample, target, isPoisoned = poisoned_trainset[0]
    # sample = reverse_normalize(sample,mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    # sample = tensor_to_PIL(sample)


    
