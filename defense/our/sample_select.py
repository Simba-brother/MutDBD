import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split
from mid_data_loader import get_class_rank

import torch.nn as nn
import torch
from commonUtils import Record
import queue
import scienceplots
import matplotlib
import matplotlib.pyplot as plt

def clean_seed(poisoned_trainset,poisoned_ids):
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
        seed_sample_id_list.extend(np.random.choice(sample_id_list, replace=False, size=10).tolist())
    clean_seedSet = Subset(poisoned_trainset,seed_sample_id_list)
    
    return clean_seedSet

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
                   poisoned_trainset,
                   poisoned_ids,
                   class_rank=None):
    '''基于模型损失值或class_rank对样本进行可疑程度排序'''
    model.to(device)
    dataset_loader = DataLoader(poisoned_trainset,batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
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

def chose_retrain_set(ranker_model, device, 
                      choice_rate, poisoned_trainset, poisoned_ids, class_rank=None):
    '''
    选择用于后门模型重训练的数据集
    '''
    ranked_sample_id_list, isPoisoned_list,loss_array = sort_sample_id(
                                                ranker_model,
                                                device,
                                                poisoned_trainset,
                                                poisoned_ids,
                                                class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    PN = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            PN += 1
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)
    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN



def draw(isPoisoned_list_1, isPoisoned_list_2 ,file_name):
    '''
    论文配图，动机章节：热力图
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isPoisoned_list]
    # 绘制热力图
    # 创建图形时设置较小的高度
    plt.style.use(['science','ieee'])
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(3, 0.5))  # 宽度为10，高度为2（可根据需要调整）
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    # plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking',fontsize='3')
    # 调整横轴刻度字号
    plt.xticks(fontsize=3)  # 明确设置横轴刻度字号为6pt
    # plt.colorbar()
    plt.yticks([])
    plt.savefig(f"imgs/sample_sort/{file_name}", bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()
    '''
    plt.style.use('science')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })
    distribution1 = [1 if flag else 0 for flag in isPoisoned_list_1]
    distribution2 = [1 if flag else 0 for flag in isPoisoned_list_2]
    
    # 创建2行1列的子图
    fig, axs = plt.subplots(2, 1, figsize=(3, 1.0))  # 总高度调整为1.0，每个子图高度约0.5

    # 确保axs是数组形式（即使只有一行）
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # 绘制第一个子图
    axs[0].imshow([distribution1], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[0].set_xlabel('Sample ranking', fontsize=3)
    axs[0].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[0].set_yticks([])

    # 绘制第二个子图
    axs[1].imshow([distribution2], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[1].set_xlabel('Sample ranking', fontsize=3)
    axs[1].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[1].set_yticks([])

    # 调整子图间距
    plt.subplots_adjust(hspace=0.3)  # 调整垂直间距

    # 保存为高分辨率图像
    plt.savefig(f"imgs/Motivation/SampleRanking/{file_name}", 
                bbox_inches='tight', 
                pad_inches=0.02,
                dpi=800,
                facecolor='white',
                edgecolor='none')

    plt.close()


if __name__ == "__main__":
    pass
