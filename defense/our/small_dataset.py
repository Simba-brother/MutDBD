import os
import random
import time
from typing import List, Tuple
from collections import defaultdict,Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR

from utils.model_eval_utils import eval_asr_acc
from utils.common_utils import convert_to_hms
from utils.dataset_utils import get_class_num
from mid_data_loader import get_backdoor_data, get_class_rank
from models.model_loader import get_model
from datasets.posisoned_dataset import get_all_dataset
from defense.our.sample_select import chose_retrain_set
from defense.our.defense_train import train



def eval_and_save(model, filtered_poisoned_testset, clean_testset, device, save_path):
    asr, acc = eval_asr_acc(model,filtered_poisoned_testset,clean_testset,device)
    torch.save(model.state_dict(), save_path)
    return asr,acc

def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          lr_scheduler=None,class_weight = None, weight_decay=None):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    if lr_scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,T_max=num_epoch,eta_min=1e-6)
    if class_weight is None:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss(class_weight.to(device))
    loss_function.to(device)
    optimal_loss = float('inf')
    best_model = model
    for epoch in range(num_epoch):
        step_loss_list = []
        for _, batch in enumerate(dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
            step_loss_list.append(loss.item())
        if lr_scheduler:
            scheduler.step()
        epoch_loss = sum(step_loss_list) / len(step_loss_list)
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            best_model = model
        print(f"epoch:{epoch},loss:{epoch_loss}")
    return model,best_model


def get_class_weights(dataset,class_num):
    label_counts = Counter()
    for sample_id in range(len(dataset)):
        label = dataset[sample_id][1]
        label_counts[label] += 1
    most_num = label_counts.most_common(1)[0][1]
    weights = []
    for cls in range(class_num):
        num = label_counts[cls]
        weights.append(round(most_num / num,1))
    return label_counts, weights


def build_clean_seedSet(poisoned_trainset,poisoned_ids):
    '''
    选择干净种子
    '''
    # 数据加载器
    poisoned_evalset_loader = DataLoader(
                poisoned_trainset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    # 获得种子
    # {class_id:[sample_id]}
    clean_sample_dict = defaultdict(list)
    label_list = []
    for _, batch in enumerate(poisoned_evalset_loader):
        Y = batch[1]
        label_list.extend(Y.tolist())

    for sample_id in range(len(poisoned_trainset)):
        if sample_id not in poisoned_ids:
            label = label_list[sample_id]
            clean_sample_dict[label].append(sample_id)

    # 获得种子数据集
    seed_sample_id_list = []
    for class_id,sample_id_list in clean_sample_dict.items():
        seed_sample_id_list.extend(random.sample(sample_id_list, 10))
    clean_seedSet = Subset(poisoned_trainset,seed_sample_id_list)
    return clean_seedSet

def sample_id_list(id_list: list[int], rate:float=0.2) -> list[int]:
    # 计算采样数量
    sample_size = int(len(id_list) * rate)
    # 从 id_list 中随机选择不放回的 20% 数据
    sampled_ids = random.sample(id_list, sample_size)
    return sampled_ids

def build_small_dataset(poisoned_trainset:DatasetFolder, poisoned_ids:list[int])->Tuple[Subset,list[int]]:
    rate = 0.2
    small_poisoned_trainset = None
    small_poisoned_ids = None
    id_list = list(range(len(poisoned_trainset)))
    # 原来数据集中的id
    sampled_ids = sample_id_list(id_list,rate)
    # 摘取的subset
    small_poisoned_trainset = Subset(poisoned_trainset, sampled_ids)
    

    subset_sample_id_list = []
    subset_poisoned_id_list = []
    # 重新映射subset的id
    for id, sample_id in enumerate(sampled_ids):
        subset_sample_id_list.append(id)
        if sample_id in poisoned_ids:
            subset_poisoned_id_list.append(id)


    print(f"原始数据集大小:{len(poisoned_trainset)}")
    print(f"原始数据集中毒样本数量:{len(poisoned_ids)}")
    print(f"数据集大小缩小比:{rate}")
    print(f"缩小数据集大小:{len(small_poisoned_trainset)}")
    print(f"缩小数据集含有的中毒样本数量:{len(subset_poisoned_id_list)}")
    return (small_poisoned_trainset,subset_poisoned_id_list)

def main_one_scence(save_dir):
    start_time = time.perf_counter()
    # 得到ranker model
    blank_model = get_model(dataset_name, model_name)
    ranker_model_state_dict = torch.load(ranker_model_state_dict_path,map_location="cpu")
    blank_model.load_state_dict(ranker_model_state_dict)
    ranker_model = blank_model

    # 构建小数据集
    small_poisoned_trainset,small_poisoned_ids = build_small_dataset(poisoned_trainset, poisoned_ids)
    
    # 得到class rank
    class_rank = get_class_rank(dataset_name,model_name,attack_name)
    # sample select
    choice_rate = 0.6
    beta = 1.0
    sigmoid_flag = False

    # 构建干净seedset
    seedSet = build_clean_seedSet(small_poisoned_trainset,small_poisoned_ids)
    select_start_time = time.perf_counter()
    # 选择样本
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = \
        chose_retrain_set(ranker_model,device,choice_rate,
                          small_poisoned_trainset,small_poisoned_ids,
                          class_rank,beta,sigmoid_flag)
    select_end_time = time.perf_counter()
    select_cost_time = select_end_time - select_start_time
    hours, minutes, seconds = convert_to_hms(select_cost_time)
    print(f"PN:{PN}")
    print(f"选择样本耗时:{hours}时{minutes}分{seconds:.1f}秒")
    # 防御重训练. 种子样本+选择的样本对模型进进下一步的 train
    availableSet = ConcatDataset([seedSet,choicedSet])
    epoch_num = 100 # 这次训练100个epoch
    lr = 1e-3
    batch_size = 512
    weight_decay=1e-3
    class_num = get_class_num(dataset_name)
    # 根据数据集中不同 class 的样本数量，设定不同 class 的 weight
    label_counter,weights = get_class_weights(availableSet, class_num)

    print(f"label_counter:{label_counter}")
    print(f"class_weights:{weights}")
    class_weights = torch.FloatTensor(weights)
    # 开始train,并返回最后一个epoch的model和在训练集上loss最小的那个best model
    train_start_time = time.perf_counter()
    last_defense_model,best_defense_model = train(
        ranker_model,device,availableSet,num_epoch=epoch_num,
        lr=lr, batch_size=batch_size,
        lr_scheduler="CosineAnnealingLR",
        class_weight=class_weights,
        weight_decay=weight_decay)
    train_end_time = time.perf_counter()
    train_cost_time = train_end_time - train_start_time
    hours, minutes, seconds = convert_to_hms(train_cost_time)
    print(f"训练耗时:{hours}时{minutes}分{seconds:.1f}秒")
    save_path = os.path.join(save_dir, "best_defense_model.pth")
    asr,acc = eval_and_save(best_defense_model, filtered_poisoned_testset, clean_testset, device, save_path)
    print(f"best_defense_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
    save_path = os.path.join(save_dir, "last_defense_model.pth")
    asr,acc = eval_and_save(last_defense_model, filtered_poisoned_testset, clean_testset, device, save_path)
    print(f"last_defense_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    print(f"one-sence总耗时:{hours}时{minutes}分{seconds:.1f}秒")

if __name__ == "__main__":

    # one-scence
    '''
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "ImageNet2012_subset"
    model_name = "DenseNet"
    attack_name = "WaNet"
    device =  torch.device("cuda:1")
    random.seed(1)
    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
    # 后门模型
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    ranker_model_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours",dataset_name,model_name,attack_name,
                                                "exp_1","best_BD_model.pth")
    save_dir = os.path.join(exp_root_dir,"small_dataset_defense_train",dataset_name,model_name,attack_name)
    os.makedirs(save_dir,exist_ok=True)
    main_one_scence(save_dir)
    '''

    # all-scence
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = [1]
    device =  torch.device("cuda:1")
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                for r_seed in r_seed_list:

                    random.seed(r_seed)
                    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
                    # 后门模型
                    if "backdoor_model" in backdoor_data.keys():
                        backdoor_model = backdoor_data["backdoor_model"]
                    else:
                        model = get_model(dataset_name, model_name)
                        state_dict = backdoor_data["backdoor_model_weights"]
                        model.load_state_dict(state_dict)
                        backdoor_model = model
                    # 训练数据集中中毒样本id
                    poisoned_ids = backdoor_data["poisoned_ids"]
                    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
                    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
                    ranker_model_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours",dataset_name,model_name,attack_name,
                                                                f"exp_{r_seed}","best_BD_model.pth")
                    save_dir = os.path.join(exp_root_dir,"small_dataset_defense_train",dataset_name,model_name,attack_name,f"exp_{r_seed}")
                    os.makedirs(save_dir,exist_ok=True)
                    print(f"{dataset_name}|{model_name}|{attack_name}|exp_{r_seed}")
                    main_one_scence(save_dir)
