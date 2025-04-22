'''
完成空白模型或后门模型的重训练
'''

import os
import time
import joblib
import copy
import numpy as np
from collections import defaultdict
from codes.ourMethod.loss import SCELoss
import matplotlib.pyplot as plt
from codes.asd.log import Record
import torch
import setproctitle
from torch.utils.data import DataLoader,Subset
import torch.nn as nn
import torch.optim as optim
from codes import config
from codes.ourMethod.defence import defence_train
from codes.scripts.dataset_constructor import *
from codes.models import get_model
from codes.common.eval_model import EvalModel
# from codes.tools import model_train_test
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


# 进程名称
proctitle = f"OMretrain|{config.dataset_name}|{config.model_name}|{config.attack_name}"
setproctitle.setproctitle(proctitle)
print(proctitle)

# 加载后门攻击配套数据
backdoor_data_path = os.path.join(config.exp_root_dir, 
                                        "ATTACK", 
                                        config.dataset_name, 
                                        config.model_name, 
                                        config.attack_name, 
                                        "backdoor_data.pth")
backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
# 后门模型
backdoor_model = backdoor_data["backdoor_model"]
poisoned_ids = backdoor_data["poisoned_ids"]
# 预制的poisoned_testset
poisoned_testset = backdoor_data["poisoned_testset"] 
# 空白模型
victim_model = get_model(dataset_name=config.dataset_name, model_name=config.model_name)

# 根据poisoned_ids得到非预制菜poisoneds_trainset和新鲜clean_testset
if config.dataset_name == "CIFAR10":
    if config.attack_name == "BadNets":
        poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = cifar10_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_WaNet()
elif config.dataset_name == "GTSRB":
    if config.attack_name == "BadNets":
        poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = gtsrb_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_WaNet()
elif config.dataset_name == "ImageNet2012_subset":
    if config.attack_name == "BadNets":
        poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = imagenet_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_WaNet()

# 数据加载器
# 打乱
poisoned_trainset_loader = DataLoader(
            poisoned_trainset, # 非预制
            batch_size=64,
            shuffle=True, # 打乱
            num_workers=4,
            pin_memory=True)
# 不打乱
poisoned_evalset_loader = DataLoader(
            poisoned_trainset, # 非预制
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
# 不打乱
clean_testset_loader = DataLoader(
            clean_testset, # 非预制
            batch_size=64, 
            shuffle=False,
            num_workers=4,
            pin_memory=True)
# 不打乱
poisoned_testset_loader = DataLoader(
            poisoned_testset,# 预制
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
# 获得设备
device = torch.device(f"cuda:{config.gpu_id}")


'''
看一下后门模型上损失情况
'''

model = backdoor_model
'''
model.to(device)
dataset_loader = poisoned_evalset_loader # 不打乱
# 损失函数
# loss_fn = SCELoss(num_classes=10, reduction="none") # nn.CrossEntropyLoss()
loss_fn = nn.CrossEntropyLoss()
loss_record = Record("loss", len(dataset_loader.dataset)) # 记录每个样本的loss
label_record = Record("label", len(dataset_loader.dataset))
model.eval()
# 判断模型是在CPU还是GPU上
for _, batch in enumerate(dataset_loader): # 分批次遍历数据加载器
    # 该批次数据
    X = batch[0].to(device)
    # 该批次标签
    Y = batch[1].to(device)
    with torch.no_grad():
        P_Y = model(X)
    loss_fn.reduction = "none" # 数据不进行规约,以此来得到每个样本的loss,而不是批次的avg_loss
    loss = loss_fn(P_Y, Y)
    loss_record.update(loss.cpu())
    label_record.update(Y.cpu())
# 基于loss排名
loss_array = loss_record.data.numpy()
# 基于loss的从小到大的样本本id排序数组
based_loss_ranked_sample_id_array =  loss_array.argsort()
# 获得对应的poisoned_flag
poisoned_flag = []
for sample_id in based_loss_ranked_sample_id_array:
    if sample_id in poisoned_ids:
        poisoned_flag.append(True)
    else:
        poisoned_flag.append(False)
# 话图看一下中毒样本在序中的分布
distribution = [1 if flag else 0 for flag in poisoned_flag]
# 绘制热力图
plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
plt.title('Heat map distribution of poisoned samples')
plt.xlabel('ranking')
plt.colorbar()
plt.yticks([])
plt.savefig("imgs/backdoor_SCEloss.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
plt.close()
'''

def module_1(model):
    model.to(device)
    dataset_loader = poisoned_evalset_loader # 不打乱
    # 损失函数
    # loss_fn = SCELoss(num_classes=10, reduction="none") # nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    loss_record = Record("loss", len(dataset_loader.dataset)) # 记录每个样本的loss
    label_record = Record("label", len(dataset_loader.dataset))
    model.eval()
    # 判断模型是在CPU还是GPU上
    for _, batch in enumerate(dataset_loader): # 分批次遍历数据加载器
        # 该批次数据
        X = batch[0].to(device)
        # 该批次标签
        Y = batch[1].to(device)
        with torch.no_grad():
            P_Y = model(X)
        loss_fn.reduction = "none" # 数据不进行规约,以此来得到每个样本的loss,而不是批次的avg_loss
        loss = loss_fn(P_Y, Y)
        loss_record.update(loss.cpu())
        label_record.update(Y.cpu())
    # 基于loss排名
    loss_array = loss_record.data.numpy()
    # 基于loss的从小到大的样本本id排序数组
    based_loss_ranked_sample_id_array =  loss_array.argsort()
    # 获得对应的poisoned_flag
    poisoned_flag = []
    for sample_id in based_loss_ranked_sample_id_array:
        if sample_id in poisoned_ids:
            poisoned_flag.append(True)
        else:
            poisoned_flag.append(False)
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in poisoned_flag]
    # 绘制热力图
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking')
    plt.colorbar()
    plt.yticks([])
    plt.savefig("imgs/retrain_CEloss_1.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()

'''
重训练
'''
# 获得种子
# {class_id:[sample_id]}
clean_sample_dict = defaultdict(list)
for sample_id, item in enumerate(poisoned_trainset):
    x = item[0]
    y = item[1]
    isPoisoned = item[2]
    if isPoisoned is False:
        clean_sample_dict[y].append(sample_id)
seed_sample_id_list = []
for class_id,sample_id_list in clean_sample_dict.items():
    seed_sample_id_list.extend(np.random.choice(sample_id_list, replace=False, size=10).tolist())
seedSet = Subset(poisoned_trainset,seed_sample_id_list)
# 基于种子retrain 5轮
def train(model,device,dataset):
    model.train()
    model.to(device)
    num_epochs = 5
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=64,
            shuffle=True, # 打乱
            num_workers=4)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for _, batch in enumerate(dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch},loss:{loss}")
    return model
model = train(model,device,seedSet)
# 看看基于损失排序后中毒样本的分布
module_1(model)


