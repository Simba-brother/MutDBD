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
import setproctitle
from torch.utils.data import DataLoader,Subset,ConcatDataset
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

def resort(ranked_sample_id_list,label_list,class_rank:list)->list:
        # 基于class_rank得到每个类别权重，原则是越可疑的类别（索引越小的类别），权（分）越大
        cls_num = len(class_rank)
        cls2score = {}
        for idx, cls in enumerate(class_rank):
            cls2score[cls] = (cls_num - idx)/cls_num 
        sample_num = len(ranked_sample_id_list)
        # 一个优先级队列
        q = queue.PriorityQueue()
        for idx, sample_id in enumerate(ranked_sample_id_list):
            sample_rank = idx+1
            sample_label = label_list[sample_id]
            cls_score = cls2score[sample_label]
            score = (sample_rank/sample_num)*cls_score
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

def train(model,device, dataset, seedSet=None, num_epoch=10,lr=1e-3):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=64,
            shuffle=True, # 打乱
            num_workers=4)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_function = nn.CrossEntropyLoss()
    # optimal_clean_acc = -1
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
        epoch_loss = sum(step_loss_list) / len(step_loss_list)
        '''
        if seedSet:
            e = EvalModel(model,seedSet,device,batch_size=8)
            acc = e.eval_acc()
            if acc > optimal_clean_acc:
                best_model = model
                optimal_clean_acc = acc
        '''
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            best_model = model
        logging.info(f"epoch:{epoch},loss:{epoch_loss}")
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



def seed_ft(model, filtered_poisoned_testset, clean_testset, seedSet, device):
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
    model = train(model,device,seedSet,lr=1e-3)
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
        device):
    '''1: 先评估一下后门模型的ASR和ACC'''
    logging.info("="*50)
    logging.info("第1步: 先评估一下后门模型的ASR和ACC")
    logging.info("="*50)
    e = EvalModel(backdoor_model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    logging.info(f"Backdoor_ASR:{asr}")
    e = EvalModel(backdoor_model,clean_testset,device)
    acc = e.eval_acc()
    logging.info(f"Backdoor_acc:{acc}")
    logging.info(f"全体中毒测试集（poisoned_testset）数据量：{len(poisoned_testset)}")
    logging.info(f"剔除了原来本属于target class的中毒测试集（filtered_poisoned_testset）数据量：{len(filtered_poisoned_testset)}")
    
    # '''1: 再评估一下ASD模型的ASR和ACC'''
    # state_dict = torch.load(config.asd_result[config.dataset_name][config.model_name][config.attack_name]["latest_model"], map_location='cpu')["model_state_dict"]
    # blank_model.load_state_dict(state_dict, strict=True)
    # e = EvalModel(blank_model,filtered_poisoned_testset,device)
    # asr = e.eval_acc()
    # print("ASD_ASR:",asr)
    # e = EvalModel(blank_model,clean_testset,device)
    # acc = e.eval_acc()
    # print("ASD_acc:",acc)
    
    '''2:先seed微调一下model'''
    logging.info("="*50)
    logging.info("第2步: 先用seed微调一下后门模型")
    logging.info("种子集是由每个类别中选择10个干净样本组成的")
    logging.info("="*50)
    freeze_model(backdoor_model,dataset_name=dataset_name,model_name=model_name)
    seed_num_epoch = 30
    seed_lr = 1e-3
    logging.info(f"种子微调的轮次为:{seed_num_epoch},学习率为:{seed_lr}")

    last_BD_model,best_BD_model = train(backdoor_model,device,dataset=seedSet,num_epoch=seed_num_epoch,lr=seed_lr)

    save_file_name = "best_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_BD_model.state_dict(), save_file_path)
    logging.info(f"基于后门模型进行种子微调后的训练损失最小的模型(best_seed_model)权重保存在:{save_file_path}")

    save_file_name = "last_BD_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_BD_model.state_dict(), save_file_path)
    logging.info(f"基于后门模型进行种子微调后的最后一轮次的模型(last_seed_model)权重保存在:{save_file_path}")

    
    '''3:评估一下种子微调后的ASR和ACC'''
    logging.info("="*50)
    logging.info("第3步: 评估一下种子微调后模型（best_seed_model）的的ASR和ACC")
    logging.info("="*50)
    e = EvalModel(best_BD_model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    logging.info(f"best_seed_model的ASR:{asr}")
    e = EvalModel(best_BD_model,clean_testset,device)
    acc = e.eval_acc()
    logging.info(f"best_seed_model的ACC:{acc}")
    
    '''4:对样本进行排序，并选择出数据集'''
    logging.info("="*50)
    logging.info("第4步: 对样本进行排序，并选择出重训练数据集")
    logging.info("="*50)
    # seed微调后排序一下样本
    class_rank = get_classes_rank(dataset_name, model_name, attack_name, config.exp_root_dir)
    ranked_sample_id_list, isPoisoned_list = sort_sample_id(
                                                best_BD_model,
                                                device,
                                                poisoned_evalset_loader,
                                                poisoned_ids,
                                                class_rank)
    choice_rate = 0.6
    num = int(len(ranked_sample_id_list)*choice_rate)
    logging.info(f"采样比例:{choice_rate},采样的数量:{num}")
    choiced_sample_id_list = ranked_sample_id_list[:num]
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            count += 1
    logging.info(f"污染样本含量:{count}/{choiced_num}")
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)

    '''5:基于种子集和选择的集对best_seed_model进行重训练'''
    logging.info("="*50)
    logging.info("第5步:基于种子集和选择的集对best_seed_model进行重训练")
    logging.info("="*50)
    # 合并种子集和选择集
    availableSet = ConcatDataset([seedSet,choicedSet])
    # 微调后门模型 
    num_epoch = 30
    lr = 1e-3
    logging.info(f"轮次为:{num_epoch},学习率为:{lr}")
    last_ft_model, best_ft_model = train(best_BD_model,device,dataset=availableSet,num_epoch=num_epoch,lr=lr)

    '''6:评估我们方法重训练后的ASR和ACC'''
    logging.info("="*50)
    logging.info("第6步:评估我们方法重训练后的ASR和ACC")
    logging.info("="*50)
    e = EvalModel(best_ft_model,filtered_poisoned_testset,device)
    asr = e.eval_acc()
    logging.info(f"我们方法重训练后的ASR:{asr}")
    e = EvalModel(best_ft_model,clean_testset,device)
    acc = e.eval_acc()
    logging.info(f"我们方法重训练后的ACC:{acc}")

    save_file_name = "best_ft_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(best_ft_model.state_dict(), save_file_path)
    logging.info(f"我们方法重训练后损失最小的模型权重保存在:{save_file_path}")

    save_file_name = "last_ft_model.pth"
    save_file_path = os.path.join(exp_dir,save_file_name)
    torch.save(last_ft_model.state_dict(), save_file_path)
    logging.info(f"我们方法重训练后最后一轮次模型权重保存在:{save_file_path}")

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



def scene_single(dataset_name, model_name, attack_name):
    # 获得实验时间戳年月日时分秒
    _time = get_formattedDateTime()
    # 随机数种子
    r_seed = 667
    np.random.seed(r_seed)
        # 进程名称
    proctitle = f"OMretrain|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    log_dir = os.path.join("log/OurMethod/defence_train/retrain",dataset_name,model_name,attack_name)
    os.makedirs(log_dir,exist_ok=True)
    log_name = f"retrain_{_time}.log"
    log_path = os.path.join(log_dir,log_name)
    logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(proctitle)
    exp_dir = os.path.join(config.exp_root_dir,"OurMethod","Retrain",dataset_name,model_name,attack_name,_time)
    os.makedirs(exp_dir,exist_ok=True)
    logging.info(f"进程名称:{proctitle}")
    logging.info(f"实验目录:{exp_dir}")
    logging.info(f"实验时间:{_time}")
    logging.info("实验主代码：codes/ourMethod/retrain.py")
    logging.info("="*50)
    logging.info("实验开始")
    logging.info("="*50)
    logging.info(f"随机数种子：{r_seed}")
    logging.info("函数：our_ft()")
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
    blank_model = get_model(dataset_name, model_name)

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
        device)
    logging.info("该实验场景结束")

if __name__ == "__main__":

    # dataset_name = config.dataset_name
    # model_name = config.model_name
    # attack_name = config.attack_name
    gpu_id = 1
    dataset_name= "ImageNet2012_subset" # CIFAR10, GTSRB, ImageNet2012_subset
    model_name= "DenseNet" # ResNet18, VGG19, DenseNet
    attack_name = "WaNet" # BadNets, IAD, Refool, WaNet
    scene_single(dataset_name, model_name, attack_name)
