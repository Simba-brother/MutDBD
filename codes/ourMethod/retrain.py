'''
完成空白模型或后门模型的重训练
'''
import logging
import sys
from codes.utils import my_excepthook
sys.excepthook = my_excepthook
from codes.common.time_handler import get_formattedDateTime
import os
import time
import joblib
import copy
import queue
import numpy as np
from collections import defaultdict
from codes.ourMethod.loss import SCELoss
import matplotlib.pyplot as plt
from codes.asd.log import Record
import torch
from torch.optim.lr_scheduler import StepLR,MultiStepLR
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
    return ranked_sample_id_list, isPoisoned_list

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
    xiter = iter(xloader) # 有监督
    uiter = iter(uloader) # 无监督
    model.train()
    batch_loss_list = []
    for batch_idx in range(kwargs["train_iteration"]):
        try:
            # 带标签中的一个批次
            xbatch = next(xiter) 
            xinput,xtarget = xbatch["img"], xbatch["target"]
        except:
            # 如果迭代器走到最后无了,则从头再来迭代
            xiter = iter(xloader)
            xbatch = next(xiter)
            xinput,xtarget = xbatch["img"], xbatch["target"]
        try:
            # 无标签batch
            ubatch = next(uiter) # 不带标签中的一个批次
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        except:
            # 如果迭代器走到最后无了,则从头再来迭代
            uiter = iter(uloader)
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, kwargs["num_classes"]).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        xinput = xinput.to(device) # 带标签批次
        xtarget = xtarget.to(device) 
        uinput1 = uinput1.to(device) # 不带标签批次
        uinput2 = uinput2.to(device)
        # uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / kwargs["temperature"])
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()


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

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)

        # 计算损失
        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / kwargs["train_iteration"],
        )
        # 半监督损失
        loss = Lx + lambda_u * Lu
        optimizer.zero_grad()
        loss.backward() # 该批次反向传播
        optimizer.step()
        batch_loss_list.append(loss.item())
    epoch_avg_loss = round(sum(batch_loss_list) / len(batch_loss_list),5)
    return epoch_avg_loss


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

def train_with_semi(model,device,seedSet,seed_sample_id_list,poisoned_trainset, poisoned_ids, poisoned_evalset_loader, class_num, num_epoch=30, lr=1e-3, batch_size=64, logger=None):
    choice_rate = 0.6
    x_id_set, u_id_set  = split_sample_id_list(model,seed_sample_id_list,poisoned_ids,poisoned_evalset_loader, choice_rate, device,logger)
    # 0表示在污染池,1表示在clean pool
    flag_list = [0] * len(poisoned_trainset)
    for i in range(len(flag_list)):
        if i in x_id_set:
            flag_list[i] = 1
    flag_array = np.array(flag_list)
    xdata = MixMatchDataset(poisoned_trainset, flag_array, labeled=True)
    udata = MixMatchDataset(poisoned_trainset, flag_array, labeled=False)
    xloader = DataLoader(xdata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    uloader = DataLoader(udata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    model.train()
    model.to(device)
    semi_criterion = MixMatchLoss(rampup_length=num_epoch, lambda_u=15) # rampup_length = 120  same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    batch_num = int(len(poisoned_trainset) / batch_size)
    semi_mixmatch = {"train_iteration": batch_num,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num}
    best_loss = float('inf')
    best_model = None
    for epoch in range(num_epoch):
        epoch_loss = semi_train_epoch(model, xloader, uloader, semi_criterion, optimizer, epoch, device, **semi_mixmatch)
        logger.info(f"Epoch:{epoch+1},loss:{epoch_loss}")
        # scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model
    return best_model,model

def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64, logger=None, use_lr_scheduer=False):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    if use_lr_scheduer:
        scheduler = MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    optimal_loss = float('inf')
    best_model = None
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
        if use_lr_scheduer:
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
    last_ft_model,best_ft_model = train(model,device,dataset,num_epoch=epoch,lr=lr,logger=logger)
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

def build_choiced_dataset(ranker_model,poisoned_trainset,poisoned_ids,poisoned_evalset_loader, choice_rate, device,logger):
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
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            count += 1
    logger.info(f"污染样本含量:{count}/{choiced_num}")
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)
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
        class_num = 0,
        logger = None):
    
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
    num_epoch = 30
    lr = 1e-3
    logger.info(f"设置重训练轮次为:{num_epoch},学习率为:{lr}")

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
    
    logger.info("trainWithSemi")
    # best_BD_model = unfreeze(best_BD_model)
    best_defense_model,last_defense_model = train_with_semi(
        best_BD_model,device,seedSet,seed_sample_id_list, poisoned_trainset, poisoned_ids, poisoned_evalset_loader,
        class_num, num_epoch=50, lr=1e-3, batch_size=128, logger=logger)

    '''5:评估我们防御后的的ASR和ACC'''
    asr, acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"防御后best_model:ASR:{asr}, ACC:{acc}")

    asr, acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    logger.info(f"防御后last_model:ASR:{asr}, ACC:{acc}")

    save_file_name = "best_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的best权重保存在:{save_file_path}")

    save_file_name = "last_defense_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_defense_model.state_dict(), save_file_path)
    logger.info(f"防御后的last权重保存在:{save_file_path}")

def get_fresh_dataset(poisoned_ids):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_BadNets()

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
    return poisoned_trainset, clean_trainset, clean_testset

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

def  scene_single(dataset_name, model_name, attack_name, r_seed=666):
    # 获得实验时间戳年月日时分秒
    _time = get_formattedDateTime()
    # 随机数种子
    np.random.seed(r_seed)
    # 进程名称
    proctitle = f"OMretrain|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    log_base_dir = "log/OurMethod/defence_train/retrain"
    log_test_dir = "log/temp"
    log_dir = os.path.join(log_test_dir,dataset_name,model_name,attack_name)
    log_file_name = f"retrain_{_time}.log"
    logger = _get_logger(log_dir,log_file_name,logger_name=_time)
    
    logger.info(proctitle)
    exp_dir = os.path.join(config.exp_root_dir,"OurMethod","Retrain",dataset_name,model_name,attack_name,_time)
    os.makedirs(exp_dir,exist_ok=True)
    logger.info(f"进程名称:{proctitle}")
    logger.info(f"实验目录:{exp_dir}")
    logger.info(f"实验时间:{_time}")
    logger.info("实验主代码：codes/ourMethod/retrain.py")
    logger.info("="*50)
    logger.info("实验开始")
    logger.info("="*50)
    logger.info(f"随机数种子：{r_seed}")
    logger.info("函数：our_ft()")
    # 加载后门攻击配套数据
    backdoor_data_path = os.path.join(config.exp_root_dir, 
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

    # 空白模型
    # mutated_model = get_model(dataset_name, model_name)
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
    poisoned_trainset, clean_trainset, clean_testset = get_fresh_dataset(poisoned_ids)
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
            poisoned_testset,# 非预制
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
    clean_testset_label_list = []
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(clean_testset)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != config.target_class_idx:
            filtered_ids.append(sample_id)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)


    # 获得设备
    device = torch.device(f"cuda:{gpu_id}")

    # 实验脚本
    our_ft(
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
        defense_model_flag = "backdoor", # str: assistant | backdoor
        logger = logger)
    
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
    #     class_num=class_num,
    #     logger = logger)
    logger.info(f"{proctitle}实验场景结束")

if __name__ == "__main__":
    
    gpu_id = 1
    r_seed = 666 # exp_1:666,exp_2:667,exp_3:668

    dataset_name= "ImageNet2012_subset" # CIFAR10, GTSRB, ImageNet2012_subset
    model_name= "DenseNet" # ResNet18, VGG19, DenseNet
    attack_name = "WaNet" # BadNets, IAD, Refool, WaNet
    class_num = 30
    scene_single(dataset_name, model_name, attack_name, r_seed=r_seed)


    # for r_seed in [666,667,668]:
    #     for dataset_name in ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
    #         for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             for attack_name in ["BadNets", "IAD", "Refool", "WaNet"]:
    #                 scene_single(dataset_name,model_name,attack_name,r_seed)
