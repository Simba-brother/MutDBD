'''
OurMethod主程序
'''
import logging
import sys
import math
from codes.utils import my_excepthook
sys.excepthook = my_excepthook
from codes.common.time_handler import get_formattedDateTime
import os
from codes.utils import convert_to_hms
import time
from collections import Counter
import joblib
import copy
import queue
import numpy as np
from collections import defaultdict
from codes.ourMethod.loss import SCELoss
import matplotlib.pyplot as plt
from codes.asd.log import Record
import torch
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR
import setproctitle
from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split
import torch.nn as nn
import torch.optim as optim
from codes import config
from codes.ourMethod.defence import defence_train
from codes.scripts.dataset_constructor import *
from codes.models import get_model
from codes.common.eval_model import EvalModel
from sklearn.model_selection import KFold
from codes.asd.loss import SCELoss, MixMatchLoss
from codes.ourMethod.loss import SimCLRLoss
from prefetch_generator import BackgroundGenerator
from itertools import cycle,islice
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
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
from torch.utils.data import Dataset

class RelabeledDataset(Dataset):
    def __init__(self, original_dataset, new_labels):
        """
        Args:
            original_dataset (Dataset): 原始数据集对象
            new_labels (list/array): 与数据集等长的新标签列表
        """
        self.original_dataset = original_dataset
        self.new_labels = new_labels
        
        # 确保新标签数量匹配
        assert len(original_dataset) == len(new_labels), \
            "标签数量必须与数据集长度一致"

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        data, _, poisoned_flag = self.original_dataset[idx]  # 假设返回格式为(data, label)
        
        # 返回新标签
        return data, self.new_labels[idx], poisoned_flag


def resort(ranked_sample_id_list,label_list,class_rank:list)->list:
        # 基于class_rank得到每个类别权重，原则是越可疑的类别（索引越小的类别），权（分）越大
        cls_num = len(class_rank)
        cls2score = {}
        for idx, cls in enumerate(class_rank):
            cls2score[cls] = (cls_num - idx)/cls_num  # 类别3：(10-0)/10 = 1, (10-9)/ 10 = 0.1
        sample_num = len(ranked_sample_id_list)
        # 一个优先级队列
        q = queue.PriorityQueue()
        for idx, sample_id in enumerate(ranked_sample_id_list):
            sample_rank = idx+1
            sample_label = label_list[sample_id]
            cls_score = cls2score[sample_label]
            score = (sample_rank/sample_num)*cls_score # cls_score 归一化了，没加log
            q.put((score,sample_id)) # 越小优先级越高，越干净
        resort_sample_id_list = []
        while not q.empty():
            resort_sample_id_list.append(q.get()[1])
        return resort_sample_id_list

def sort_sample_id(model,
                   device,
                   poisoned_evalset_loader,
                   poisoned_ids,
                   class_rank=None):
    '''基于模型损失值或class_rank对样本进行可疑程度排序'''
    model.to(device)
    dataset_loader = poisoned_evalset_loader # 不打乱
    # 损失函数
    # loss_fn = SCELoss(num_classes=10, reduction="none") # nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
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
    based_loss_ranked_sample_id_list =  loss_array.argsort().tolist()
    
    if class_rank is None:
        ranked_sample_id_list = based_loss_ranked_sample_id_list
    else:
        label_list = label_record.data.numpy().tolist()
        ranked_sample_id_list  = resort(based_loss_ranked_sample_id_list,label_list,class_rank)
    # 获得对应的poisoned_flag
    isPoisoned_list = []
    for sample_id in ranked_sample_id_list:
        if sample_id in poisoned_ids:
            isPoisoned_list.append(True)
        else:
            isPoisoned_list.append(False)
    return ranked_sample_id_list, isPoisoned_list,loss_array

def draw(isPoisoned_list,file_name):
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isPoisoned_list]
    # 绘制热力图
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking')
    plt.colorbar()
    plt.yticks([])
    plt.savefig(f"imgs/{file_name}", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()

def train_epoch(model,dataset_loader,loss_fn,device,optimizer):
    model.train()
    step_loss_list = []
    for _, batch in enumerate(dataset_loader):
        optimizer.zero_grad()
        X = batch[0].to(device)
        Y = batch[1].to(device)
        P_Y = model(X)
        loss = loss_fn(P_Y, Y)
        loss.backward()
        optimizer.step()
        step_loss_list.append(loss.item())
    epoch_loss = sum(step_loss_list) / len(step_loss_list)
    return model, epoch_loss
    

def train_dynamic(model,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader, num_epoch=30, lr=1e-3, batch_size=64, logger=None):
    # model = unfreeze(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    best_loss = float('inf')
    choice_rate = 0.8
    pre_choiced_sample_id_list = None
    for epoch in range(num_epoch):
        choicedSet,choiced_sample_id_list = build_choiced_dataset(model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
        if pre_choiced_sample_id_list:
            inter_num = len(set(choiced_sample_id_list) & set(pre_choiced_sample_id_list))
            logger.info(f"和上次选择的训练集的交集数量:{inter_num}") 
        availableSet = ConcatDataset([seedSet,choicedSet])
        dataset_loader = DataLoader(availableSet,batch_size=batch_size,shuffle=True,num_workers=4)
        model,epoch_loss = train_epoch(model,dataset_loader,loss_function,device,optimizer)
        scheduler.step()
        logger.info(f"epoch:{epoch},loss:{epoch_loss}")
        pre_choiced_sample_id_list = choiced_sample_id_list
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
    return best_model, model



def train_KFold(model_o,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader, num_epoch=30, lr=1e-3, batch_size=64, logger=None):
    choice_rate = 0.6
    choicedSet,choiced_sample_id_list = build_choiced_dataset(model_o,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
    availableSet = ConcatDataset([seedSet,choicedSet])
    # K折交叉验证
    kf = KFold(n_splits=5,shuffle=True,random_state=666)
    # 用于存储每折的性能指标
    max_acc = 0
    best_model = None
    for fold, (train_idx, test_idx) in enumerate(kf.split(availableSet)):
        train_subset = Subset(availableSet,train_idx)
        val_subset = Subset(availableSet,test_idx)
        # 创建DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        # 初始模型
        model = copy.deepcopy(model_o)
        # 初始优化器
        optimizer = optim.Adam(model.parameters(),lr=lr)
        # 初始lr调整器
        scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
        # 初始化损失函数
        loss_function = nn.CrossEntropyLoss()
        # 训练模型
        model.train()
        model.to(device)
        for epoch in range(num_epoch):
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                X = batch[0].to(device)
                Y = batch[1].to(device)
                P_Y = model(X)
                loss = loss_function(P_Y, Y)
                loss.backward()
                optimizer.step()
        # 模型评估
        em = EvalModel(model,val_subset,device,batch_size=batch_size)
        acc = em.eval_acc()
        logger.info(f"Fold:{fold+1},eval_ACC:{acc}")
        if acc > max_acc:
            max_acc = acc
            best_model = model
    return best_model,model


def train_with_eval(model,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader, num_epoch=30, lr=1e-3, batch_size=64, logger=None):
    choice_rate = 0.6
    choicedSet,choiced_sample_id_list = build_choiced_dataset(model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
    availableSet = ConcatDataset([seedSet,choicedSet])
    max_acc = 0
    best_model = None
    train_loader = DataLoader(availableSet, batch_size=batch_size, shuffle=True)
    # 初始优化器
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # 初始lr调整器
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    # 初始化损失函数
    loss_function = nn.CrossEntropyLoss()
    # 训练模型
    model.train()
    model.to(device)
    for epoch in range(num_epoch):
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
        # 每个轮次的模型评估
        em = EvalModel(model,seedSet,device,batch_size=batch_size)
        acc = em.eval_acc()
        logger.info(f"Epoch:{epoch+1},eval_ACC:{acc}")
        if acc > max_acc:
            max_acc = acc
            best_model = model
        scheduler.step()
    return best_model,model


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]

def semi_train_epoch(model, xloader, uloader, criterion, optimizer, epoch, device, **kwargs):
    total_start_time = time.perf_counter()
    cycle_ready_start_time = time.perf_counter()
    xiter = cycle(xloader) # 有监督
    uiter = cycle(uloader) # 无监督
    xlimited_cycled_data = islice(xiter,0,kwargs["train_iteration"])
    ulimited_cycled_data = islice(uiter,0,kwargs["train_iteration"])
    model.train()
    if kwargs["amp"]:
        scaler = GradScaler()
    cycle_ready_end_time = time.perf_counter()    
    cycle_cost_time = cycle_ready_end_time - cycle_ready_start_time
    # batch_loss_list = []
    sum_batch_loss = 0.0
    batch_num = 0
    data_cost_time = 0
    data_stage_1_cost_time = 0
    data_stage_2_cost_time = 0
    data_stage_3_cost_time = 0
    dava_mv_cost_time = 0
    model_cost_time = 0
    append_cost_time = 0
    enumerate_cost_time = 0
    enumerate_start_time = time.perf_counter()
    # torch.cuda.synchronize() 
    for batch_idx,(xbatch,ubatch) in enumerate(zip(xlimited_cycled_data,ulimited_cycled_data)):
        batch_num += 1
        '''数据段开始'''
        data_start_time = time.perf_counter()
        xinput, xtarget = xbatch["img"], xbatch["target"]
        uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        
        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, kwargs["num_classes"]).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        data_mv_start_time = time.perf_counter()
        # 数据上设备
        xinput = xinput.to(device)
        xtarget = xtarget.to(device) 
        uinput1 = uinput1.to(device)
        uinput2 = uinput2.to(device)
        data_mv_end_time = time.perf_counter()
        dava_mv_cost_time += (data_mv_end_time - data_mv_start_time)
        # uinput2 = uinput2.cuda(gpu, non_blocking=True)
        data_stage_1_end_time = time.perf_counter()
        data_stage_1_cost_time += (data_stage_1_end_time - data_start_time)
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / kwargs["temperature"])
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()
        data_stage_2_end_time = time.perf_counter()
        data_stage_2_cost_time += (data_stage_2_end_time - data_stage_1_end_time)

        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(kwargs["alpha"], kwargs["alpha"])
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        data_stage_3_end_time = time.perf_counter()
        data_stage_3_cost_time += (data_stage_3_end_time - data_stage_2_end_time)
        data_end_time = time.perf_counter()
        data_cost_time += (data_end_time - data_start_time)
        '''数据段结束'''
        '''模型段开始'''
        model_start_time = time.perf_counter()
        optimizer.zero_grad()
        if kwargs["amp"]:
            with autocast():
                logit = [model(mixed_input[0])]
                for input in mixed_input[1:]:
                    logit.append(model(input))

                # put interleaved samples back
                logit = interleave(logit, batch_size)
                xlogit = logit[0]
                ulogit = torch.cat(logit[1:], dim=0)
                # 计算损失
                '''
                class_weight = torch.FloatTensor([0.1,0.1,0.1,0.3,0.1,0.1,0.1,0.1,0.1,0.1])
                class_weight = class_weight.to(device)
                '''
                Lx, Lu, lambda_u = criterion(
                    xlogit,
                    mixed_target[:batch_size],
                    ulogit,
                    mixed_target[batch_size:],
                    epoch + batch_idx / kwargs["train_iteration"],
                    # class_weight = class_weight
                )
                # 半监督损失
                loss = Lx + lambda_u * Lu
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logit = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logit.append(model(input))
            # put interleaved samples back
            logit = interleave(logit, batch_size)
            xlogit = logit[0]
            ulogit = torch.cat(logit[1:], dim=0)
            # 计算损失
            '''
            class_weight = torch.FloatTensor([0.1,0.1,0.1,0.3,0.1,0.1,0.1,0.1,0.1,0.1])
            class_weight = class_weight.to(device)
            '''
            Lx, Lu, lambda_u = criterion(
                xlogit,
                mixed_target[:batch_size],
                ulogit,
                mixed_target[batch_size:],
                epoch + batch_idx / kwargs["train_iteration"],
                # class_weight = class_weight
            )
            # 半监督损失
            loss = Lx + lambda_u * Lu
            loss.backward() # 该批次反向传播
            optimizer.step()
        '''模型段结束'''
        model_end_time = time.perf_counter()
        model_cost_time += (model_end_time - model_start_time)
        '''append开始'''
        append_start_time = time.perf_counter()
        # batch_loss_list.append(loss.item())
        sum_batch_loss += loss
        append_end_time = time.perf_counter()
        append_cost_time += (append_end_time - append_start_time)
    enumerate_end_time = time.perf_counter()
    enumerate_cost_time = enumerate_end_time - enumerate_start_time - (model_cost_time+data_cost_time+append_cost_time)
    # epoch_avg_loss = round(sum(batch_loss_list) / len(batch_loss_list),5)
    avg_start_time = time.perf_counter()
    epoch_avg_loss = round(sum_batch_loss.item() / batch_num, 5) 
    avg_end_time = time.perf_counter()
    avg_cost_time = avg_end_time - avg_start_time
    total_end_time = time.perf_counter()
    total_cost_time =  total_end_time - total_start_time

    time_dict = {
        "total_cost_time":total_cost_time,
        "cycle_cost_time":cycle_cost_time,
        "data_cost_time":data_cost_time,
        "model_cost_time":model_cost_time,
        "append_cost_time":append_cost_time,
        "enumerate_cost_time":enumerate_cost_time,
        "avg_cost_time":avg_cost_time,
        "data_stage_1_cost_time":data_stage_1_cost_time,
        "data_stage_2_cost_time":data_stage_2_cost_time,
        "data_stage_3_cost_time":data_stage_3_cost_time,
        "data_mv_cost_time":dava_mv_cost_time
    }
    return epoch_avg_loss,time_dict

class MixMatchDataset(Dataset):
        """Semi-supervised MixMatch dataset.

        Args:
            dataset (Dataset): The dataset to be wrapped.
            semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
            labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
                set (default: True).
        """

        def __init__(self, dataset, semi_idx, labeled=True):
            super(MixMatchDataset, self).__init__()
            # self.dataset = copy.deepcopy(dataset)
            # self.dataset_2 = copy.deepcopy(dataset_2)
            self.dataset = dataset
            # self.dataset_2 = dataset_2
            if labeled:
                # 有标签的情况，从semi_id array中找到对应的索引
                # 比如arr = np.array([1,0,1,0])
                # np.nonzero(arr==1)[0]就为np.array([0,2]),self.semi_indice,__len__类的魔术方法就使用这个
                # np.nonzero(arr==0)[0]就为np.array([1,3])
                self.semi_indice = np.nonzero(semi_idx == 1)[0]
            else:
                self.semi_indice = np.nonzero(semi_idx == 0)[0]
            self.labeled = labeled
            # self.prefetch = self.dataset.prefetch
            # if self.prefetch:
            #     self.mean, self.std = self.dataset.mean, self.dataset.std

        def __getitem__(self, index):
            # index in [0,len(self.semi_indice)-1]
            if self.labeled:
                item1 = self.dataset[self.semi_indice[index]] # self.semi_indice[index] = sampl_id(datset)
                img = item1[0]
                target = item1[1]
                item = {}
                item["img"] = img
                item["target"] = target
                item["labeled"] = True
            else:
                item1 = self.dataset[self.semi_indice[index]]
                item2 = self.dataset[self.semi_indice[index]]
                # item2 = self.dataset_2[self.semi_indice[index]]
                img1 = item1[0]
                img2 = item2[0]
                item = {}
                item["img1"] = img1
                item["img2"] = img2
                item["target"] = item1[1]
                item["labeled"] = False
            return item

        def __len__(self):
            # 这里的semi_indice其实就时选择出的带标签或不带标签的样本索引array
            return len(self.semi_indice)

def train_with_subset(model,device,logger,
                      choice_rate,epoch_num=120,lr=1e-3,batch_size=64):
    clean_trainset, clean_testset = get_cleanTrainSet_cleanTestSet("CIFAR10","WaNet")
    label_num = int(len(clean_trainset)*choice_rate)
    id_list = list(range(len(clean_trainset)))
    choiced_id_list = random.sample(id_list, label_num)
    subset = Subset(clean_trainset, choiced_id_list)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    # 初始优化器
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # 初始lr调整器
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    # 初始化损失函数
    loss_function = nn.CrossEntropyLoss()
    # 训练模型
    model.train()
    model.to(device)
    max_acc = 0
    for epoch in range(epoch_num):
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
        # 每个轮次的模型评估
        em = EvalModel(model,clean_testset,device,batch_size=batch_size)
        acc = em.eval_acc()
        logger.info(f"Epoch:{epoch+1},clean_ACC:{acc}")
        if acc > max_acc:
            max_acc = acc
            best_model = model
            best_epoch = epoch
        scheduler.step()
    logger.info(f"Best Epoch:{best_epoch}")
    return best_model,model

class PrefetchDataLoader(DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def train_with_semi(choice_model,retrain_model,device,seed_sample_id_list,poisoned_trainset, poisoned_ids, poisoned_evalset_loader, class_num, logger,
                    choice_rate=0.6, epoch_num=120, batch_num = 1024, lr=2e-3, batch_size=64):
    # 基于选择模型的loss值切分数据集
    x_id_set, u_id_set  = split_sample_id_list(choice_model,seed_sample_id_list,poisoned_ids,poisoned_evalset_loader, choice_rate, device,logger)
    # 0表示在污染池,1表示在clean pool
    flag_list = [0] * len(poisoned_trainset)
    for x_id in x_id_set:
        flag_list[x_id] = 1

    flag_array = np.array(flag_list)
    train_set = poisoned_trainset
    model = retrain_model
    xdata = MixMatchDataset(train_set, flag_array, labeled=True)
    udata = MixMatchDataset(train_set, flag_array, labeled=False)


    xloader = DataLoader(xdata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    uloader = DataLoader(udata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    # xloader = PrefetchDataLoader(xdata, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory=True)
    # uloader = PrefetchDataLoader(udata, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory=True)
    model.train()
    model.to(device)
    semi_criterion = MixMatchLoss(rampup_length=epoch_num, lambda_u=15) # rampup_length = 120  same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    amp = False
    semi_mixmatch = {"train_iteration": batch_num,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num, "amp":amp}
    best_loss = float('inf')
    best_model = None
    for epoch in range(epoch_num):
        epoch_loss,time_dict = semi_train_epoch(model, xloader, uloader, semi_criterion, optimizer, epoch, device, **semi_mixmatch)
        th,tm,ts = convert_to_hms(time_dict["total_cost_time"])
        logger.info(f"Epoch:{epoch+1},loss:{epoch_loss},CostTime:{th}:{tm}:{ts:.3f}")
        '''
        ch,cm,cs = convert_to_hms(time_dict["cycle_cost_time"])
        dh,dm,ds = convert_to_hms(time_dict["data_cost_time"])
        ds1h,ds1m,ds1s = convert_to_hms(time_dict["data_stage_1_cost_time"]) 
        mvh,mvm,mvs = convert_to_hms(time_dict["data_mv_cost_time"]) 
        ds2h,ds2m,ds2s = convert_to_hms(time_dict["data_stage_2_cost_time"]) 
        ds3h,ds3m,ds3s = convert_to_hms(time_dict["data_stage_3_cost_time"]) 
        mh,mm,ms = convert_to_hms(time_dict["model_cost_time"])
        aph,apm,aps = convert_to_hms(time_dict["append_cost_time"])
        eh,em,es = convert_to_hms(time_dict["enumerate_cost_time"]) 
        avh,avm,avs = convert_to_hms(time_dict["avg_cost_time"]) 
        logger.info(f"EpochTime:{th}:{tm}:{ts:.3f}")
        logger.info(f"CycleTime:{ch}:{cm}:{cs:.3f}")
        logger.info(f"DataTime:{dh}:{dm}:{ds:.3f}")
        logger.info(f"\tDS1:{ds1h}:{ds1m}:{ds1s:.3f}")
        logger.info(f"\t\tMv:{mvh}:{mvm}:{mvs:.3f}")
        logger.info(f"\tDS2:{ds2h}:{ds2m}:{ds2s:.3f}")
        logger.info(f"\tDS3:{ds3h}:{ds3m}:{ds3s:.3f}")
        logger.info(f"ModelTime:{mh}:{mm}:{ms:.3f}")
        logger.info(f"AppendTime:{aph}:{apm}:{aps:.3f}")
        logger.info(f"EnumTime:{eh}:{em}:{es:.3f}")
        logger.info(f"AvgTime:{avh}:{avm}:{avs:.3f}")
        '''

        # scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
    return best_model,model

def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          logger=None,lr_scheduler=None, class_weight = None, weight_decay=None):
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
        logger.info(f"epoch:{epoch},loss:{epoch_loss}")
    return model,best_model

def train_oneFold(model,device, dataset, seedSet=None, num_epoch=30, lr=1e-3, batch_size=64, logger=None, use_lr_scheduer=False):
    model.train()
    model.to(device)
    total_size = len(dataset)
    train_size = int(0.8*len(dataset))
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset = ConcatDataset([seedSet,val_dataset])
    train_dataset_loader = DataLoader(
            train_dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    if use_lr_scheduer:
        scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model = None
    for epoch in range(num_epoch):
        step_loss_list = []
        for _, batch in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
            step_loss_list.append(loss.item())
        if use_lr_scheduer:
            scheduler.step()
        e = EvalModel(model,val_dataset,device,batch_size=32)
        val_acc = e.eval_acc()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
        logger.info(f"epoch:{epoch},val_acc:{val_acc}")
    return model,best_model

def get_classes_rank(dataset_name, model_name, attack_name, exp_root_dir)->list:
    '''获得类别排序'''
    mutated_rate = 0.01
    measure_name = "Precision_mean"
    if dataset_name in ["CIFAR10","GTSRB"]:
        grid = joblib.load(os.path.join(exp_root_dir,"grid.joblib"))
        classes_rank = grid[dataset_name][model_name][attack_name][mutated_rate][measure_name]["class_rank"]
    elif dataset_name == "ImageNet2012_subset":
        classRank_data = joblib.load(os.path.join(
            exp_root_dir,
            "ClassRank",
            dataset_name,
            model_name,
            attack_name,
            str(mutated_rate),
            measure_name,
            "ClassRank.joblib"
        ))
        classes_rank =classRank_data["class_rank"]
    else:
        raise Exception("数据集名称错误")
    return classes_rank

def get_classes_rank_v2(exp_root_dir,dataset_name,model_name,attack_name):
    data_path = os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,"res.joblib")
    data = joblib.load(data_path)
    return data["class_rank"]

def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def freeze_model(model,dataset_name,model_name):
    if dataset_name == "CIFAR10" or dataset_name == "GTSRB":
        if model_name == "ResNet18":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'features.5' in name or 'features.4' in name or 'features.3' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'dense4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    elif dataset_name == "ImageNet2012_subset":
        if model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name,param in model.named_parameters():
                if 'classifier' in name or 'features.denseblock4' in name or 'features.denseblock3' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "ResNet18":
            for name,param in model.named_parameters():
                if 'fc' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    else:
        raise Exception("模型不存在")
    return model

def seed_ft(model, filtered_poisoned_testset, clean_testset, seedSet, device,logger):
    # FT前模型评估
    e = EvalModel(model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    print("backdoor_ASR:",asr)
    e = EvalModel(model,clean_testset,device)
    acc = e.eval_acc()
    print("backdoor_acc:",acc)
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(model)
    draw(isPoisoned_list,file_name="backdoor_loss.png")
    # 冻结
    freeze_model(model,dataset_name=dataset_name,model_name=model_name)
    # 获得class_rank
    class_rank = get_classes_rank()
    # 基于种子集和后门模型微调10轮次
    model = train(model,device,seedSet,lr=1e-3,logger=logger)
    e = EvalModel(model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    print("FT_ASR:",asr)
    e = EvalModel(model,clean_testset,device)
    acc = e.eval_acc()
    print("FT_acc:",acc)
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(model)
    draw(isPoisoned_list,file_name="retrain10epoch_loss.png")
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(model,class_rank)
    draw(isPoisoned_list,file_name="retrain10epoch_lossAndClassRank.png")

def ft(model,device,dataset,epoch,lr,logger):
    '''
    微调
    '''
    # 冻结
    freeze_model(model,dataset_name=dataset_name,model_name=model_name)
    last_ft_model,best_ft_model = train(model,device,dataset,num_epoch=epoch,lr=lr,
                                        logger=logger)
    return last_ft_model,best_ft_model

def eval_asr_acc(model,poisoned_set,clean_set,device):
    e = EvalModel(model,poisoned_set,device)
    asr = e.eval_acc()
    e = EvalModel(model,clean_set,device)
    acc = e.eval_acc()
    return asr,acc

def our_ft(
        backdoor_model,
        poisoned_testset,
        filtered_poisoned_testset, 
        clean_testset,
        seedSet,
        exp_dir,
        poisoned_ids,
        poisoned_trainset,
        poisoned_evalset_loader,
        device,
        assistant_model = None,
        defense_model_flag = "backdoor",
        logger = None):
    '''1: 先评估一下后门模型的ASR和ACC'''
    logger.info("="*50)
    logger.info("第1步: 先评估一下后门模型的ASR和ACC")
    logger.info("="*50)
    asr,acc = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"后门模型的ASR:{asr},后门模型的ACC:{acc}")

    '''评估一下变异模型的ASR和ACC'''
    if assistant_model:
        asr,acc = eval_asr_acc(assistant_model,filtered_poisoned_testset,clean_testset,device)
        logger.info(f"辅助样本选择模型的: ASR:{asr}, ACC:{acc}")


    logger.info(f"全体中毒测试集（poisoned_testset）数据量：{len(poisoned_testset)}")
    logger.info(f"剔除了原来本属于target class的中毒测试集（filtered_poisoned_testset）数据量：{len(filtered_poisoned_testset)}")
    
    # '''1: 再评估一下ASD模型的ASR和ACC'''
    # state_dict = torch.load(config.asd_result[config.dataset_name][config.model_name][config.attack_name]["latest_model"], map_location='cpu')["model_state_dict"]
    # blank_model.load_state_dict(state_dict, strict=True)
    # e = EvalModel(blank_model,filtered_poisoned_testset,device)
    # asr = e.eval_acc()
    # print("ASD_ASR:",asr)
    # e = EvalModel(blank_model,clean_testset,device)
    # acc = e.eval_acc()
    # print("ASD_acc:",acc)
    
    '''2:种子微调模型'''
    logger.info("="*50)
    logger.info("第2步: 种子微调模型")
    logger.info("种子集: 由每个类别中选择10个干净样本组成")
    logger.info("="*50)

    seed_num_epoch = 30
    seed_lr = 1e-3
    logger.info(f"种子微调轮次:{seed_num_epoch},学习率:{seed_lr}")
    if assistant_model:
        logger.info('种子微调辅助模型')
        last_ft_assistent_model, best_ft_assitent_model = ft(assistant_model,device,seedSet,seed_num_epoch,seed_lr,logger=logger)
        
        save_file_name = "best_ft_assistant_model.pth"
        save_file_path = os.path.join(exp_dir,save_file_name)
        torch.save(best_ft_assitent_model.state_dict(), save_file_path)
        logger.info(f"基于辅助模型进行种子微调后的best模型权重保存在:{save_file_path}")

        save_file_name = "last_ft_assistent_model.pth"
        save_file_path = os.path.join(exp_dir,save_file_name)
        torch.save(last_ft_assistent_model.state_dict(), save_file_path)
        logger.info(f"基于辅助模型进行种子微调后的last模型权重保存在:{save_file_path}")

    logger.info('种子微调后门模型')
    last_BD_model,best_BD_model = ft(backdoor_model,device,seedSet,seed_num_epoch,seed_lr,logger=logger)
 
    logger.info("保存种子微调后门模型")
    save_file_name = "best_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的训练损失最小的模型权重保存在:{save_file_path}")

    save_file_name = "last_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的最后一轮次的模型(last_seed_model)权重保存在:{save_file_path}")

    
    '''3:评估一下种子微调后的ASR和ACC'''
    logger.info("="*50)
    logger.info("第3步: 评估一下种子微调后模型的的ASR和ACC")
    logger.info("="*50)
    if assistant_model:
        asr,acc = eval_asr_acc(best_ft_assitent_model,filtered_poisoned_testset,clean_testset,device)
        logger.info(f"基于辅助种子微调后的: ASR:{asr}, ACC:{acc}")
    asr,acc = eval_asr_acc(best_BD_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"基于后门模型种子微调后的: ASR:{asr}, ACC:{acc}")

    '''4:对样本进行排序，并选择出数据集'''
    logger.info("="*50)
    logger.info("第4步: 对样本进行排序，并选择出重训练数据集")
    logger.info("="*50)
    # seed微调后排序一下样本
    class_rank = get_classes_rank(dataset_name, model_name, attack_name, config.exp_root_dir)
    if assistant_model:
        ranker_model = best_ft_assitent_model
        logger.info(f"排序辅助模型:变异模型")
    else:
        ranker_model = best_BD_model
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(
                                                ranker_model,
                                                device,
                                                poisoned_evalset_loader,
                                                poisoned_ids,
                                                class_rank)
    choice_rate = 0.7
    num = int(len(ranked_sample_id_list)*choice_rate)
    logger.info(f"采样比例:{choice_rate},采样的数量:{num}")
    choiced_sample_id_list = ranked_sample_id_list[:num]
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            count += 1
    logger.info(f"污染样本含量:{count}/{choiced_num}")
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)

    '''5:基于种子集和选择的集对微调后的后门模型或辅助模型进行重训练'''
    logger.info("="*50)
    logger.info("第5步:基于种子集和选择的集对best_seed_model进行重训练")
    logger.info("="*50)
    # 合并种子集和选择集
    availableSet = ConcatDataset([seedSet,choicedSet])
    # 微调后门模型 
    num_epoch = 50
    lr = 1e-3
    logger.info(f"轮次为:{num_epoch},学习率为:{lr}")
    if assistant_model and defense_model_flag == "assistant":
        logger.info(f"防御模型:变异模型")
        last_defense_model, best_defense_model = train(best_ft_assitent_model,device,dataset=availableSet,num_epoch=num_epoch,lr=lr,logger=logger)
    else:
        logger.info(f"防御模型:后门模型")
        batch_size = 512
        last_defense_model, best_defense_model = train(best_BD_model,device,dataset=availableSet,num_epoch=num_epoch,lr=lr, batch_size=batch_size, logger=logger, use_lr_scheduer=True)
        # last_defense_model, best_defense_model = train_oneFold(best_BD_model,device,dataset=availableSet,seedSet=seedSet, num_epoch=num_epoch,lr=lr, batch_size=batch_size, logger=logger, use_lr_scheduer=False)
        
    '''6:评估我们防御后的的ASR和ACC'''
    logger.info("="*50)
    logger.info("第6步:评估我们防御后的的ASR和ACC")
    logger.info("="*50)

    asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"防御后bestmodel:ASR:{asr}, ACC:{acc}")

    asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"防御后lastmodel:ASR:{asr}, ACC:{acc}")

    save_file_name = "best_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的best权重保存在:{save_file_path}")

    save_file_name = "last_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的last权重保存在:{save_file_path}")



def visualization_sampling(ranked_sample_id_list:list, poisoned_id_list:list):

    # 记录排名分布，1表示对应该排名位置的样本被采样了，反之不然。例如，如果rank_distribution[0]=1则说明排名第一的样本被采样了。
    rank_distribution = [0]*len(ranked_sample_id_list)
    # 遍历每个位次
    for rank in range(len(ranked_sample_id_list)):
        # 当前位次的样本索引（item）
        item = ranked_sample_id_list[rank]
        # 判断当前位次样本在不在采样list中
        if item in poisoned_id_list:
            # 如果该位次的样本被采样了
            rank_distribution[rank] = 1
    # 绘制热力图
    plt.imshow([rank_distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('Position Index')
    plt.colorbar()
    plt.yticks([])
    plt.savefig("imgs/sampleing_rank_distribution_ClassRank.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()

def build_choiced_dataset(ranker_model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader, choice_rate, device,logger):
    # seed微调后排序一下样本
    # class_rank = get_classes_rank(dataset_name, model_name, attack_name, config.exp_root_dir)
    class_rank = get_classes_rank_v2(config.exp_root_dir,dataset_name, model_name, attack_name)
    ranked_sample_id_list, isPoisoned_list,loss_array = sort_sample_id(
                                                ranker_model,
                                                device,
                                                poisoned_evalset_loader,
                                                poisoned_ids,
                                                class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            count += 1
    logger.info(f"污染样本含量:{count}/{choiced_num}")
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)

    # visualization_sampling(ranked_sample_id_list,poisoned_ids)
    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list

def split_sample_id_list(ranker_model,seed_sample_id_list,poisoned_ids,poisoned_evalset_loader, choice_rate, device,logger):
    # seed微调后排序一下样本
    class_rank = get_classes_rank(dataset_name, model_name, attack_name, config.exp_root_dir)
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(
                                                ranker_model,
                                                device,
                                                poisoned_evalset_loader,
                                                poisoned_ids,
                                                class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]

    seed_set = set(seed_sample_id_list)
    choice_set = set(choiced_sample_id_list)
    x_set = seed_set.union(choice_set)
    u_set = set(remain_sample_id_list)

    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            count += 1
    logger.info(f"污染样本含量:{count}/{choiced_num}")
    return x_set, u_set


def get_class_weights(dataset):
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



class SelfSupervisedDataset(Dataset):
        """Semi-supervised MixMatch dataset.

        Args:
            dataset (Dataset): The dataset to be wrapped.
            semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
            labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
                set (default: True).
        """

        def __init__(self, dataset):
            super(SelfSupervisedDataset, self).__init__()
            self.dataset = dataset
        def __getitem__(self, index):
            item1 = self.dataset[index]
            item2 = self.dataset[index]
            img1 = item1[0]
            img2 = item2[0]
            item = {}
            item["img1"] = img1
            item["img2"] = img2
            item["target"] = item1[1]
            return item

        def __len__(self):
            # 这里的semi_indice其实就时选择出的带标签或不带标签的样本索引array
            return len(self.dataset)


def purifing_feature_extractor(model,dataset,device, epoch, batch_size,amp,logger):
    dataset = SelfSupervisedDataset(dataset)
    dataset_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )
    
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.4)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    criterion = SimCLRLoss(temperature=0.5)
    model.train()
    if amp:
        scaler = GradScaler()

    for epoch in range(epoch):
        for batch_idx, batch in enumerate(dataset_loader):
            # batch train前优化器先梯度清0
            optimizer.zero_grad() 
            img1, img2 = batch["img1"], batch["img2"]
            data = torch.cat([img1.unsqueeze(1), img2.unsqueeze(1)], dim=1)
            b, c, h, w = img1.size()
            data = data.view(-1, c, h, w)
            data = data.cuda(device, non_blocking=True)
            if amp:
                with (): # 开启自动精度转换
                    output = model(data).view(b, 2, -1)
                    loss = criterion(output)
                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update() # scaler factor更新
            else:
                output = model(data).view(b, 2, -1)
                loss = criterion(output)
                loss.backward()
                optimizer.step()
        scheduler.step()
        logger.info(f'Epoch:{epoch+1}, 当前学习率为:{optimizer.param_groups[0]["lr"]}')
    return model





def cut_off_discussion(
        backdoor_model,
        poisoned_testset,
        filtered_poisoned_testset, 
        clean_testset,
        seedSet,
        seed_sample_id_list,
        exp_dir,
        poisoned_ids,
        poisoned_trainset,
        poisoned_evalset_loader,
        device,
        class_num,
        logger,
        blank_model = None):
    
    # last_model, best_model = train(blank_model,device,poisoned_trainset,num_epoch=100,
    #     lr=1e-3, batch_size=512, logger=logger, 
    #     lr_scheduler="CosineAnnealingLR",
    #     class_weight=None,weight_decay=1e-3)
    # asr, acc = eval_asr_acc(best_model,filtered_poisoned_testset,clean_testset,device)
    # print("best_model,asr:",asr)
    # print("best_model,acc:",acc)

    # asr, acc = eval_asr_acc(last_model,filtered_poisoned_testset,clean_testset,device)
    # print("last_model,asr:",asr)
    # print("last_model,acc:",acc)
    
    logger.info("cut_off_discussion")
    '''1: 先评估一下后门模型的ASR和ACC'''
    logger.info("第1步: 先评估一下后门模型的ASR和ACC")
    asr,acc = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"后门模型的ASR:{asr},后门模型的ACC:{acc}")

    logger.info(f"剔除了原来本属于target class的中毒测试集（filtered_poisoned_testset）数据量：{len(filtered_poisoned_testset)}")

    '''2:种子微调模型'''
    logger.info("第2步: 种子微调模型")
    logger.info("种子集: 由每个类别中选择10个干净样本组成")
    seed_num_epoch = 30
    seed_lr = 1e-3
    logger.info(f"种子微调轮次:{seed_num_epoch},学习率:{seed_lr}")

    last_BD_model,best_BD_model = ft(backdoor_model,device,seedSet,seed_num_epoch,seed_lr,logger=logger)
 
    logger.info("保存种子微调后门模型")
    save_file_name = "best_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的训练损失最小的模型权重保存在:{save_file_path}")

    save_file_name = "last_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的最后一轮次的模型(last_seed_model)权重保存在:{save_file_path}")

    
    '''3:评估一下种子微调后的ASR和ACC'''
    
    logger.info("第3步: 评估一下种子微调后模型的的ASR和ACC")
    asr,acc = eval_asr_acc(best_BD_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"基于后门模型种子微调后的: ASR:{asr}, ACC:{acc}")

    '''4:重训练'''
    
    logger.info("朴素的retrain")
    # 解冻
    # best_BD_model = unfreeze(best_BD_model)
    for choice_rate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        # 每个采样率都是从微调好的原始模型copy出来的，目的是保持使用同样的best_BD_model
        finetuned_model = copy.deepcopy(best_BD_model)
        logger.info(f"Cut off:{choice_rate}")
        # 使用finetuned_model去选择样本
        choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list = build_choiced_dataset(
            finetuned_model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
        availableSet = ConcatDataset([seedSet,choicedSet])
        # finetuned_model = unfreeze(finetuned_model)
        epoch_num = 100
        lr = 1e-3
        batch_size = 512
        weight_decay=1e-3
        label_counter,weights = get_class_weights(availableSet)
        logger.info(f"label_counter:{label_counter}")
        logger.info(f"class_weights:{weights}")
        class_weights = torch.FloatTensor(weights)
        last_defense_model,best_defense_model = train(
            finetuned_model,device,availableSet,num_epoch=epoch_num,
            lr=lr, batch_size=batch_size, logger=logger, 
            lr_scheduler="CosineAnnealingLR",
            class_weight=None,weight_decay=weight_decay)
        asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
        logger.info(f"朴素监督防御后best_model:ASR:{asr}, ACC:{acc}")
        asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
        logger.info(f"朴素监督防御后last_model:ASR:{asr}, ACC:{acc}")
        save_file_name = "best_defense_model.pth"
        _dir =  os.path.join(exp_dir,str(choice_rate))
        os.makedirs(_dir,exist_ok=True)
        save_file_path = os.path.join(_dir,save_file_name)
        torch.save(best_defense_model.state_dict(), save_file_path)
        logger.info(f"朴素监督防御后的best权重保存在:{save_file_path}")
        save_file_name = "last_defense_model.pth"
        save_file_path = os.path.join(_dir,save_file_name)
        torch.save(last_defense_model.state_dict(), save_file_path)
        logger.info(f"朴素监督防御后的last权重保存在:{save_file_path}")

def our_ft_2(
        backdoor_model,
        poisoned_testset,
        filtered_poisoned_testset, 
        clean_testset,
        seedSet,
        seed_sample_id_list,
        exp_dir,
        poisoned_ids,
        poisoned_trainset,
        poisoned_evalset_loader,
        device,
        class_num,
        logger,
        blank_model = None):
    
    '''1: 先评估一下后门模型的ASR和ACC'''
    logger.info("第1步: 先评估一下后门模型的ASR和ACC")
    asr,acc = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"后门模型的ASR:{asr},后门模型的ACC:{acc}")

    logger.info(f"全体中毒测试集（poisoned_testset）数据量：{len(poisoned_testset)}")
    logger.info(f"剔除了原来本属于target class的中毒测试集（filtered_poisoned_testset）数据量：{len(filtered_poisoned_testset)}")

    '''2:种子微调模型'''
    logger.info("第2步: 种子微调模型")
    logger.info("种子集: 由每个类别中选择10个干净样本组成")
    seed_num_epoch = 30
    seed_lr = 1e-3
    logger.info(f"种子微调轮次:{seed_num_epoch},学习率:{seed_lr}")

    last_BD_model,best_BD_model = ft(backdoor_model,device,seedSet,seed_num_epoch,seed_lr,logger=logger)
 
    logger.info("保存种子微调后门模型")
    save_file_name = "best_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的训练损失最小的模型权重保存在:{save_file_path}")

    save_file_name = "last_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_BD_model.state_dict(), save_file_path)
    logger.info(f"基于后门模型进行种子微调后的最后一轮次的模型(last_seed_model)权重保存在:{save_file_path}")

    
    '''3:评估一下种子微调后的ASR和ACC'''
    
    logger.info("第3步: 评估一下种子微调后模型的的ASR和ACC")
    asr,acc = eval_asr_acc(best_BD_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"基于后门模型种子微调后的: ASR:{asr}, ACC:{acc}")

    '''4:重训练'''
    
    logger.info("朴素的retrain")
    # 解冻
    # best_BD_model = unfreeze(best_BD_model)
    choice_rate = 0.6
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list = build_choiced_dataset(best_BD_model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
    availableSet = ConcatDataset([seedSet,choicedSet])
    epoch_num = 100
    lr = 1e-3
    batch_size = 512
    weight_decay=1e-3

    label_counter,weights = get_class_weights(availableSet)
    logger.info(f"label_counter:{label_counter}")
    logger.info(f"class_weights:{weights}")
    class_weights = torch.FloatTensor(weights)
    last_defense_model,best_defense_model = train(
        best_BD_model,device,availableSet,num_epoch=epoch_num,
        lr=lr, batch_size=batch_size, logger=logger, 
        lr_scheduler="CosineAnnealingLR",
        class_weight=class_weights,weight_decay=weight_decay)
    asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"朴素监督防御后best_model:ASR:{asr}, ACC:{acc}")
    asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"朴素监督防御后last_model:ASR:{asr}, ACC:{acc}")
    save_file_name = "best_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_defense_model.state_dict(), save_file_path)
    logger.info(f"朴素监督防御后的best权重保存在:{save_file_path}")
    save_file_name = "last_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_defense_model.state_dict(), save_file_path)
    logger.info(f"朴素监督防御后的last权重保存在:{save_file_path}")
    
    # logger.info("train_dynamic_choice")
    # best_defense_model, last_defense_model = train_dynamic(
    #     best_BD_model,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader,
    #     num_epoch=num_epoch,lr=lr,batch_size=128,logger=logger)

    # logger.info("KFold")
    # best_defense_model,last_defense_model = train_KFold(
    #     best_BD_model,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader,
    #     num_epoch=30, lr=1e-3, batch_size=64, logger=logger)

    # logger.info("trainWithEval")
    # best_defense_model,last_defense_model = train_with_eval(
    #     best_BD_model,device,seedSet,poisoned_trainset, poisoned_ids, poisoned_evalset_loader,
    #     num_epoch=30, lr=1e-3, batch_size=64, logger=logger)

    # trainwithsubset
    # logger.info("trainwithsubset")
    # model = best_BD_model
    # model = unfreeze(model)
    # best_defense_model, last_defense_model = train_with_subset(model,device,logger,
    #                   choice_rate=0.6,epoch_num=120,lr=1e-3,batch_size=64)
    
    # 半监督
    '''
    logger.info("半监督训练-开始")
    choice_model = best_BD_model # 使用种子微调后的模型作为选择模型
    if blank_model is None:
        # best_BD_model = unfreeze(best_BD_model)
        retrain_model = best_BD_model
    else:
        logger.info("retrain空白模型")
        retrain_model = blank_model
    
    # 使用自监督方法纯化特征提取器
    # 先将retrain model进行自监督学习1000个epoch,旨在进行纯化模型的特征提取器
    # logger.info("纯化特征训练-开始")
    # retrain_model = purifing_feature_extractor(
    #     model = retrain_model,
    #     dataset = poisoned_trainset,
    #     device=device, 
    #     epoch = 1000, 
    #     batch_size = 512,
    #     amp = True,
    #     logger=logger)
    # logger.info("纯化特征训练-结束")

    # 使用半监督训练
    epoch_num = 120
    batch_size = 512
    batch_num = math.ceil(1024 * 64 / batch_size)
    lr = 2e-3
    
    choice_rate = 0.6

    logger.info(f"设置重训练轮次为:{epoch_num},学习率为:{lr}")

    best_defense_model,last_defense_model = train_with_semi(
        choice_model,retrain_model,device,seed_sample_id_list,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,
        class_num,logger,choice_rate=choice_rate, epoch_num=epoch_num, batch_num=batch_num, lr=lr, batch_size=batch_size)
    logger.info("半监督训练-结束")

    
    asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"半监督防御后best_model:ASR:{asr}, ACC:{acc}")

    asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"半监督防御后last_model:ASR:{asr}, ACC:{acc}")
    '''

    '''
    # 半监督+再监督
    logger.info("半监督+再监督训练模式")
    choice_model = best_BD_model # 使用种子微调后的模型作为选择模型
    if blank_model is None:
        # best_BD_model = unfreeze(best_BD_model)
        logger.info("防御模型：SeedFT_Best_model")
        retrain_model = best_BD_model
    else:
        logger.info("防御模型：空白模型")
        retrain_model = blank_model
    # 使用半监督训练
    epoch_num = 120
    batch_size = 512
    batch_num = math.ceil(1024 * 64 / batch_size)
    lr = 2e-3
    logger.info(f"半监督:epoch:{epoch_num}|lr:{lr}|batch_size:{batch_size}|batch_num/epoch:{batch_num}")
    
    choice_rate = 0.6
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list = build_choiced_dataset(choice_model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,choice_rate,device,logger)
    availableSet = ConcatDataset([seedSet,choicedSet])

    
    logger.info("半监督训练(stage1)-开始")
    best_stage1_model,last_stage1_model = train_with_semi(
        choice_model,retrain_model,device,seed_sample_id_list,poisoned_trainset,poisoned_ids,poisoned_evalset_loader,
        class_num,logger,choice_rate=choice_rate, epoch_num=epoch_num, batch_num=batch_num, lr=lr, batch_size=batch_size)
    logger.info("半监督训练-结束")

    
    # logger.info("直接加载半监督训练结果")
    # stage_1_model = retrain_model
    # stage_1_model.load_state_dict(torch.load("/data/mml/backdoor_detect/experiments/OurMethod_new/ImageNet2012_subset/ResNet18/BadNets/exp_11/semi120+sft40/best_defense_model.pth",map_location="cpu"))
    # stage_1_model.to(device)
    
    stage_1_model = best_stage1_model

    asr, acc = eval_asr_acc(stage_1_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"半监督防御后best_model:ASR:{asr}, ACC:{acc}")
    asr, acc = eval_asr_acc(stage_1_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"半监督防御后last_model:ASR:{asr}, ACC:{acc}")

    save_file_name = "best_stage1_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_stage1_model.state_dict(), save_file_path)
    logger.info(f"防御后的best权重保存在:{save_file_path}")
    save_file_name = "last_stage1_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_stage1_model.state_dict(), save_file_path)
    logger.info(f"防御后的last权重保存在:{save_file_path}")


    em = EvalModel(stage_1_model,remainSet,device)
    confidence_list = em.get_confidence_list()
    conf_array = np.array(confidence_list)
    remain_sample_id_list
    k = max(1,math.ceil(len(conf_array)*0.3))
    sorted_indices = np.argsort(conf_array)[::-1] # 降序
    pseudo_local_ids = sorted_indices[:k].tolist()
    pseudo_sample_ids = [remain_sample_id_list[p_i] for p_i in pseudo_local_ids]
    intersection = set(pseudo_sample_ids) & set(poisoned_ids)
    logger.info(f"伪标签数据中的中毒样本含量:{len(intersection)}/{len(pseudo_sample_ids)}") 
    pseudo_dataset = Subset(remainSet,pseudo_local_ids)
    em2 = EvalModel(stage_1_model, pseudo_dataset, device)
    pred_label_list = em2.get_pred_labels()
    pseudo_dataset = RelabeledDataset(pseudo_dataset, pred_label_list)
    scp_dataset = ConcatDataset([seedSet,choicedSet,pseudo_dataset])
    
    stage2_dataset = scp_dataset
    # 再监督微调一下(stage2_dataset)
    epoch_num = 30
    lr = 1e-3
    batch_size = 512
    weight_decay=1e-5
    label_counter,weights = get_class_weights(stage2_dataset)
    logger.info(f"epoch_num:{epoch_num}|lr:{lr}|batch_size:{batch_size}|weight_decay:{weight_decay}")
    logger.info(f"label_counter:{label_counter}")
    logger.info(f"class_weights:{weights}")
    class_weights = torch.FloatTensor(weights)
    last_defense_model,best_defense_model = train(
        stage_1_model,
        device,
        stage2_dataset,
        num_epoch=epoch_num,
        lr=lr,
        batch_size=batch_size,
        logger=logger,
        lr_scheduler="CosineAnnealingLR",
        class_weight=class_weights,
        weight_decay=weight_decay)
    asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"再监督防御后best_model:ASR:{asr}, ACC:{acc}")
    asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"再监督防御后last_model:ASR:{asr}, ACC:{acc}")
    save_file_name = "best_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的best权重保存在:{save_file_path}")
    save_file_name = "last_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的last权重保存在:{save_file_path}")
    '''

def get_fresh_dataset(poisoned_ids):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            poisoned_trainset, poisoned_testset_noTargetClass, clean_trainset, clean_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            # clean_trainset, clean_testset = cifar10_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = cifar10_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_WaNet()
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = gtsrb_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_WaNet()
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = imagenet_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_WaNet()
    return poisoned_trainset, poisoned_testset_noTargetClass, clean_trainset, clean_testset

def _get_logger(log_dir,log_file_name,logger_name):
    # 创建一个logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,log_file_name)

    # logger的文件处理器，包括日志等级，日志路径，模式，编码等
    file_handler = logging.FileHandler(log_path,mode="w",encoding="UTF-8")
    file_handler.setLevel(logging.DEBUG)

    # logger的格式化处理器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    #将格式化器添加到文件处理器
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志对象中
    logger.addHandler(file_handler)
    return logger
'''
def try_semi_train(model,train_set,device,logger,class_num,
                   label_rate = 0.6,batch_size = 64,epoch_num = 120,batch_num = 1024,lr = 2e-3):
    
    # 尝试一下半监督训练
    
    # ==划分数据集==
    label_num = int(len(train_set)*label_rate)
    id_list = list(range(len(train_set)))
    choiced_id_list = random.sample(id_list, label_num)
    # 0表示在ulabel,1表示在xlabel
    flag_list = [0] * len(train_set)
    for choiced_id in choiced_id_list:
        flag_list[choiced_id] = 1
    flag_array = np.array(flag_list)
    xdata = MixMatchDataset(train_set, flag_array, labeled=True)
    udata = MixMatchDataset(train_set, flag_array, labeled=False)
    xloader = DataLoader(xdata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    uloader = DataLoader(udata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    # ==训练==
    # 模型训练模式
    model.train()
    model.to(device)
    # 模型参数优化器
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # 半监督损失函数
    semi_criterion = MixMatchLoss(rampup_length=epoch_num, lambda_u=75) # rampup_length = 120  same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    # scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    # 半监督参数配置
    semi_mixmatch = {"train_iteration": batch_num,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num}
    best_loss = float('inf')
    best_model = None
    for epoch in range(epoch_num):
        epoch_loss = semi_train_epoch(model, xloader, uloader, semi_criterion, optimizer, epoch, device, **semi_mixmatch)
        logger.info(f"Epoch:{epoch+1},loss:{epoch_loss}")
        # scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
    return best_model,model
'''

def scene_single(dataset_name, model_name, attack_name, r_seed):
    start_time = time.perf_counter()
    # 获得实验时间戳年月日时分秒
    _time = get_formattedDateTime()
    # 随机数种子
    np.random.seed(r_seed)
    # 进程名称
    proctitle = f"OMretrain|{dataset_name}|{model_name}|{attack_name}|{r_seed}"
    setproctitle.setproctitle(proctitle)
    log_base_dir = "log/cut_off"
    # log_base_dir = "log/temp"
    log_dir = os.path.join(log_base_dir,dataset_name,model_name,attack_name)
    log_file_name = f"retrain_r_seed_{r_seed}_{_time}.log"
    logger = _get_logger(log_dir,log_file_name,logger_name=_time)
    
    logger.info(proctitle)
    exp_dir = os.path.join(config.exp_root_dir,"cut_off",dataset_name,model_name,attack_name,f"exp_{r_seed}")
    os.makedirs(exp_dir,exist_ok=True)
    logger.info(f"进程名称:{proctitle}")
    logger.info(f"实验目录:{exp_dir}")
    logger.info(f"实验时间:{_time}")
    logger.info("实验主代码：codes/ourMethod/retrain.py/scene_single")
    logger.info("实验开始")
    logger.info(f"随机数种子:{r_seed}")
    # 加载后门攻击配套数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK",
        dataset_name,
        model_name,
        attack_name,
        "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
    # 后门模型
    backdoor_model = backdoor_data["backdoor_model"]
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 预制的poisoned_testset
    # poisoned_testset = backdoor_data["poisoned_testset"] 
    
    # 空白模型
    blank_model = get_model(dataset_name, model_name)
    
    
    # 某个变异模型
    # mutations_dir = os.path.join(
    #     config.exp_root_dir,
    #     "MutationModels",
    #     dataset_name,
    #     model_name,
    #     attack_name
    # )
    # mutate_rate = 0.05
    # m_id = 3
    # logger.info(f"变异率:{mutate_rate}, id:{m_id}")
    # mutated_model.load_state_dict(torch.load(os.path.join(mutations_dir,str(mutate_rate),"Gaussian_Fuzzing",f"model_{m_id}.pth")))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset和新鲜clean_testset
    poisoned_trainset, poisoned_testset_noTargetClass, clean_trainset, clean_testset = get_fresh_dataset(poisoned_ids)
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
                poisoned_trainset,
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
                poisoned_testset_noTargetClass,# 预制
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
        seed_sample_id_list.extend(np.random.choice(sample_id_list, replace=False, size=10).tolist())
    seedSet = Subset(poisoned_trainset,seed_sample_id_list)

    # 从poisoned_testset中剔除原来就是target class的数据
    # clean_testset_label_list = []
    # for _, batch in enumerate(clean_testset_loader):
    #     Y = batch[1]
    #     clean_testset_label_list.extend(Y.tolist())
    # filtered_ids = []
    # for sample_id in range(len(clean_testset)):
    #     sample_label = clean_testset_label_list[sample_id]
    #     if sample_label != config.target_class_idx:
    #         filtered_ids.append(sample_id)
    # filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)


    # 获得设备
    device = torch.device(f"cuda:{gpu_id}")
    '''
    # 实验脚本
    # our_ft(
    #     backdoor_model,
    #     poisoned_testset,
    #     filtered_poisoned_testset, 
    #     clean_testset,
    #     seedSet,
    #     exp_dir,
    #     poisoned_ids,
    #     poisoned_trainset,
    #     poisoned_evalset_loader,
    #     device,
    #     assistant_model = None,
    #     defense_model_flag = "backdoor", # str: assistant | backdoor
    #     logger = logger)
    '''
    cut_off_discussion(
        backdoor_model,
        poisoned_testset_noTargetClass,
        poisoned_testset_noTargetClass, 
        clean_testset,
        seedSet,
        seed_sample_id_list,
        exp_dir,
        poisoned_ids,
        poisoned_trainset,
        poisoned_evalset_loader,
        device,
        class_num,
        logger,
        blank_model = None)
    # our_ft_2(
    #     backdoor_model,
    #     poisoned_testset,
    #     filtered_poisoned_testset, 
    #     clean_testset,
    #     seedSet,
    #     seed_sample_id_list,
    #     exp_dir,
    #     poisoned_ids,
    #     poisoned_trainset,
    #     poisoned_evalset_loader,
    #     device,
    #     class_num,
    #     logger,
    #     blank_model = None)
    logger.info(f"{proctitle}实验场景结束")
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    logger.info(f"共耗时:{hours}时{minutes}分{seconds:.3f}秒")


def get_cleanTrainSet_cleanTestSet(dataset_name, attack_name):
    clean_trainset = None
    clean_testset = None
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            clean_trainset, clean_testset = cifar10_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = cifar10_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = cifar10_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = cifar10_WaNet()
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            clean_trainset, clean_testset = gtsrb_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = gtsrb_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = gtsrb_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = gtsrb_WaNet()
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            clean_trainset, clean_testset = imagenet_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = imagenet_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = imagenet_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = imagenet_WaNet()
    return clean_trainset, clean_testset


def get_backdoor(dataset_name,model_name,attack_name):
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK",
        dataset_name,
        model_name,
        attack_name,
        "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
    # 后门模型
    backdoor_model = backdoor_data["backdoor_model"]
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 预制的poisoned_testset
    poisoned_testset = backdoor_data["poisoned_testset"] 
    return backdoor_model, poisoned_ids, poisoned_testset

'''
def try_semi_train_main(dataset_name, model_name, attack_name, class_num, r_seed):
    # 获得实验时间戳年月日时分秒
    _time = get_formattedDateTime()
    # 随机数种子
    np.random.seed(r_seed)
    # 进程名称
    proctitle = f"SemiTrain_Test|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    log_test_dir = "log/temp"
    log_dir = os.path.join(log_test_dir,dataset_name,model_name,attack_name)
    log_file_name = f"semiTrain_{_time}.log"
    logger = _get_logger(log_dir,log_file_name,logger_name=_time)
    logger.info(proctitle)
    # 获得数据集
    clean_trainset, clean_testset = get_cleanTrainSet_cleanTestSet(dataset_name,attack_name)
    # 获得空白模型
    # model = get_model(dataset_name, model_name)
    # 获得后门模型
    backdoor_model, poisoned_ids, poisoned_testset = get_backdoor(dataset_name, model_name, attack_name)
    model = backdoor_model
    device = torch.device("cuda:1")
    # 开始半监督训练
    best_model,last_model = try_semi_train(model,clean_trainset,device,logger,class_num,
                   label_rate = 0.6,batch_size = 64,epoch_num = 120,batch_num = 1024,lr = 2e-3)
    # 评估
    e = EvalModel(best_model,clean_testset,device)
    acc = e.eval_acc()
    logger.info(f"bast_model ACC:{acc}",acc)
    e = EvalModel(last_model,clean_testset,device)
    acc = e.eval_acc()
    logger.info(f"last_model ACC:{acc}")
'''

def get_classNum(dataset_name):
    class_num = None
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    return class_num

if __name__ == "__main__":
    
    gpu_id = 0
    r_seed = 2
    dataset_name= "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
    model_name= "ResNet18" # ResNet18, VGG19, DenseNet
    attack_name ="BadNets" # BadNets, IAD, Refool, WaNet
    class_num = get_classNum(dataset_name)

    # try_semi_train_main(dataset_name, model_name, attack_name, class_num, r_seed)
    scene_single(dataset_name, model_name, attack_name, r_seed=r_seed)

    
    # gpu_id = 0
    # r_seed = 11
    # dataset_name = "CIFAR10"
    # class_num = get_classNum(dataset_name)
    # model_name = "DenseNet"
    # for attack_name in ["IAD"]:
    #     scene_single(dataset_name,model_name,attack_name,r_seed)
    

    # gpu_id = 1
    # for r_seed in [1,2,3,4,5]:
    #     for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
    #         class_num = get_classNum(dataset_name)
    #         for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             for attack_name in ["BadNets", "IAD", "Refool", "WaNet"]:
    #                 scene_single(dataset_name,model_name,attack_name,r_seed)