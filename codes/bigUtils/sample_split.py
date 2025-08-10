import queue
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Record(object):
    '''
    分批次的记录数据
    '''
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = torch.zeros(self.size)

    def update(self, batch_data):
        self.data[self.ptr : self.ptr + len(batch_data)] = batch_data
        self.ptr += len(batch_data)

def resort(ranked_sample_id_list,label_list,class_rank:list)->list:
    '''
    基于loss排好序的序列，外加考虑class rank信息对样本进行排序
    '''
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

def sort_sample_id(model, device, dataset_loader, class_rank=None):
    '''
    基于样本loss或class rank对样本进行排序
    '''
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    # 记录每个样本的loss
    loss_record = Record("loss", len(dataset_loader.dataset))
    # 记录每个样本的label
    label_record = Record("label", len(dataset_loader.dataset))
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
        ranked_sample_id_list = based_loss_ranked_sample_id_list # 朴素基于损失值从小到大排序样本id
    else:
        label_list = label_record.data.numpy().tolist()
        ranked_sample_id_list  = resort(based_loss_ranked_sample_id_list,label_list,class_rank)
    '''
    # 获得对应的poisoned_flag
    isPoisoned_list = []
    for sample_id in ranked_sample_id_list:
        if sample_id in poisoned_ids:
            isPoisoned_list.append(True)
        else:
            isPoisoned_list.append(False)
    '''
    return ranked_sample_id_list


def _split(
        ranker_model, # 用于划分的模型，通常是种子微调后的模型
        poisoned_trainset,
        poisoned_ids,
        device,
        class_rank = None,
        choice_rate = 0.5 # cut_off
        ):
    # 数据集加载器
    poisoned_trainset_loader = DataLoader(
        poisoned_trainset,
        batch_size = 256,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    # 样本排序
    ranked_sample_id_list = sort_sample_id(
                            ranker_model,
                            device,
                            poisoned_trainset_loader,
                            class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    # 干净池总数
    choiced_num = len(choiced_sample_id_list)
    # 干净池中的中毒数量
    p_count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            p_count += 1
    # 干净池的中毒比例
    poisoning_rate = round(p_count/choiced_num, 4)
    return p_count, choiced_num, poisoning_rate