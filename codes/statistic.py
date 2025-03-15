'''
统计中毒数据的信息
'''
import os
import torch
import config
from codes.scripts.dataset_constructor import *

# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_poisoned_dataset as gtsrb_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_poisoned_dataset as gtsrb_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_poisoned_dataset as gtsrb_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_poisoned_dataset as gtsrb_WaNet_gen_poisoned_dataset



# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset as imagenet_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_poisoned_dataset as imagenet_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_poisoned_dataset as imagenet_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_poisoned_dataset as imagenet_WaNet_gen_poisoned_dataset

# transform数据集
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets, gtsrb_IAD, gtsrb_Refool, gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets, imagenet_IAD, imagenet_Refool, imagenet_WaNet

exp_root_dir = config.exp_root_dir
dataset_name = "ImageNet2012_subset" # "CIFAR10","GTSRB","ImageNet2012_subset"
model_name =  "ResNet18" # "ResNet18","VGG19","DenseNet"
attack_name = "BadNets" # "BadNets", "IAD", "Refool", "WaNet"
target_class_idx = config.target_class_idx

backdoor_data_path = os.path.join(exp_root_dir,"ATTACK",dataset_name,model_name,attack_name,"backdoor_data.pth")
backdoor_data = torch.load(backdoor_data_path, map_location="cpu")

poisoned_trainset = backdoor_data["poisoned_trainset"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_testset = backdoor_data["poisoned_testset"]

targetClass_trainset = ExtractTargetClassDataset(poisoned_trainset, target_class_idx)
purePoisoned_trainset = ExtractDatasetByIds(poisoned_trainset,poisoned_ids)

print(f"训练集数量:{len(poisoned_trainset)}")
print(f"测试集数量:{len(poisoned_testset)}")

print(f"target class样本数量:{len(targetClass_trainset)}")
rate = round(len(purePoisoned_trainset)/len(targetClass_trainset),3)
print(f"攻击类别中木马样本比例:{rate}")





