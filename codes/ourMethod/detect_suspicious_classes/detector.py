'''
怀疑集检测器
'''
import os
import queue
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from sklearn.metrics import classification_report

from codes import config
# 得到格式化时间串
from codes.common.time_handler import get_formattedDateTime
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids
from codes.utils import entropy,priorityQueue_2_list,calcu_LCR
from codes.common.logging_handler import get_Logger


'''
=======核心函数区==================
'''


def detect_by_LCR_model(
        df:pd.DataFrame,
        class_num:int,
        mutated_model_global_id_list:list[int],
        stat_name="mean"):
    '''
    通过LCR排名进行怀疑集检测
    '''    
    '''
    dict(list[int])
    {class_id:[measure_1,..,]}
    '''
    class_list_dict = defaultdict(list)
    for class_i in range(class_num):
        # 过滤出当前类df
        class_df = df.loc[df["GT_label"]==class_i]
        backdoor_model_preLabel_list =  list(class_df["original_backdoorModel_preLabel"])
        for i in mutated_model_global_id_list:
            preLabel_list = list(class_df[f"model_{i}"])
            # 计算该变异模型的LCR
            LCR_model = calcu_LCR(backdoor_model_preLabel_list,preLabel_list)
            class_list_dict[class_i].append(LCR_model)
    # 基于熵值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        LCR_model_list = class_list_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(np.mean(LCR_model_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(LCR_model_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),2)
    return classes_rank,target_class_ranking_percent

def detect_by_LCR_sample(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list,stat_name="mean"):
    '''
    通过LCR排名进行怀疑集检测
    '''
    # 变异模型global_id_list
    '''
    dict(list[int])
    {class_id:[measure_1,..,]}
    '''
    class_list_dict = defaultdict(list)
    for class_i in range(class_num):
        # 过滤出当前类df
        class_df = df.loc[df["GT_label"]==class_i]
        for _,row in class_df.iterrows():
            label_o = row["original_backdoorModel_preLabel"]
            count = 0
            for i in mutated_model_global_id_list:
                if row[f"model_{i}"] != label_o:
                    count += 1 
            # 变异模型集在当前样本上预测标签的LCR
            LCR_sample = round(count/len(mutated_model_global_id_list),4)
            class_list_dict[class_i].append(LCR_sample)
    # 基于熵值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        LCR_sample_list = class_list_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(np.mean(LCR_sample_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(LCR_sample_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),2)
    return classes_rank,target_class_ranking_percent


def detect_by_entropy_model(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list[int],stat_name="mean"):
    '''
    通过entropy排名进行怀疑集检测
    '''
    '''
    dict(list[int])
    {class_id:[measure_1,..,]}
    '''
    class_list_dict = defaultdict(list)
    for class_i in range(class_num):
        # 过滤出当前类df
        class_df = df.loc[df["GT_label"]==class_i]
        for i in mutated_model_global_id_list:
            preLabel_list = class_df[f"model_{i}"]
            # 计算该变异模型的熵
            e_model = entropy(preLabel_list)
            class_list_dict[class_i].append(e_model)
    # 基于熵值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        e_model_list = class_list_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(np.mean(e_model_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(e_model_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),2)
    return classes_rank,target_class_ranking_percent


def detect_by_entropy_sample(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list[int],stat_name="mean"):
    '''
    通过entropy排名进行怀疑集检测
    '''
    '''
    dict(list[int])
    {class_id:[measure_1,..,]}
    '''
    class_list_dict = defaultdict(list)
    for class_i in range(class_num):
        # 过滤出当前类df
        class_df = df.loc[df["GT_label"]==class_i]
        for _,row in class_df.iterrows():
            pre_labels = []
            for i in mutated_model_global_id_list:
                pre_labels.append(row[f"model_{i}"])
            # 变异模型集在当前样本上预测标签的熵值
            e_sample = entropy(pre_labels)
            class_list_dict[class_i].append(e_sample)
    # 基于熵值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        e_sample_list = class_list_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(np.mean(e_sample_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(e_sample_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),2)
    return classes_rank,target_class_ranking_percent


def detect_by_loss(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list[int],stat_name="mean"):
    '''
    通过loss排名进行怀疑集检测
    '''
    '''
    通过precision排名进行怀疑集检测
    '''
    # 变异模型global_id_list
    '''
    dict(list[int])
    {class_id:[measure_1,..,]}
    '''
    class_list_dict = defaultdict(list)
    for class_i in range(class_num):
        # 过滤出当前类df
        class_df = df.loc[df["GT_label"]==class_i]
        for i in mutated_model_global_id_list:
            ceLoss_list = class_df[f"model_{i}"]
            class_list_dict[class_i].extend(ceLoss_list)
    # 基于均值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        measure_list = class_list_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(sum(measure_list)/len(measure_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(measure_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),2)
    return classes_rank,target_class_ranking_percent

def detect_by_precision(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list[int],stat_name = "mean"):
    '''
    通过precision排名进行怀疑集检测
    '''
    # 变异模型global_id_list
    gt_label_list = df["GT_label"]
    '''
    dict(list[int])
    {class_id:[precision_1,..,]}
    '''
    class_precisionList_dict = defaultdict(list)
    for i in mutated_model_global_id_list:
        preLabel_list = df[f"model_{i}"]
        report = classification_report(gt_label_list,preLabel_list,output_dict=True,zero_division=0)
        for class_i in range(class_num):
            measure = report[str(class_i)]["precision"]
            class_precisionList_dict[class_i].append(measure)
    # 基于均值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        measure_list = class_precisionList_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(sum(measure_list)/len(measure_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(measure_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),3)
    return classes_rank,target_class_ranking_percent

def detect_by_recall(df:pd.DataFrame,class_num:int,mutated_model_global_id_list:list[int],stat_name="mean"):
    '''
    通过recall排名进行怀疑集检测
    '''

    gt_label_list = df["GT_label"]
    '''
    dict(list[int])
    {class_id:[precision_1,..,]}
    '''
    class_precisionList_dict = defaultdict(list)
    for i in mutated_model_global_id_list:
        preLabel_list = df[f"model_{i}"]
        report = classification_report(gt_label_list,preLabel_list,output_dict=True,zero_division=0)
        for class_i in range(class_num):
            measure = report[str(class_i)]["recall"]
            class_precisionList_dict[class_i].append(measure)
    # 基于均值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(class_num):
        measure_list = class_precisionList_dict[class_i]
        if stat_name == "mean":
            stat_measure = round(sum(measure_list)/len(measure_list),4)
        if stat_name == "var":
            stat_measure = round(np.var(measure_list),4)
        item = (stat_measure,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),3)
    return classes_rank,target_class_ranking_percent

'''
========普通功能函数区============
'''




'''
=======数据加载区=========
'''
def load_df(dataset_name,model_name,attack_name,mutated_rate,df_name:str):
    if df_name == "preLabel":
        df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                dataset_name,
                model_name,
                attack_name,
                str(mutated_rate),
                "preLabel.csv")
        )
    elif df_name == "CELoss":
        df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                dataset_name,
                model_name,
                attack_name,
                str(mutated_rate),
                "CELoss.csv")
        )
    return df

'''
========数据保存区============
'''

'''
==========结果展示区==============
'''


'''
=======主函数区======
'''

def detect_method_pool(
        df_Label:pd.DataFrame,
        df_CELoss:pd.DataFrame,
        class_num:int,
        mutated_model_global_id_list:list[int],
        method:str):
    if method == "Precision_avg":
        class_rank,rank_rate = detect_by_precision(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Precision_var":
        class_rank,rank_rate = detect_by_precision(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Loss_avg":
        class_rank,rank_rate = detect_by_loss(df_CELoss,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Loss_var":
        class_rank,rank_rate = detect_by_loss(df_CELoss,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Recall_avg":
        class_rank,rank_rate = detect_by_recall(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Recall_var":
        class_rank,rank_rate = detect_by_recall(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Entropy_model_mean":
        class_rank,rank_rate = detect_by_entropy_model(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Entropy_model_var":
        class_rank,rank_rate = detect_by_entropy_model(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Entropy_sample_mean":
        class_rank,rank_rate = detect_by_entropy_sample(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Entropy_sample_var":
        class_rank,rank_rate = detect_by_entropy_sample(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "LCR_model_mean":
        class_rank,rank_rate = detect_by_LCR_model(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "LCR_model_var":
        class_rank,rank_rate = detect_by_LCR_model(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "LCR_sample_mean":
        class_rank,rank_rate = detect_by_LCR_sample(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "LCR_sample_var":
        class_rank,rank_rate = detect_by_LCR_sample(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    return class_rank,rank_rate


def main():

    data = {}
    dataset_name_list = config.cur_dataset_name_list
    model_name_list = config.cur_model_name_list
    attack_name_list = config.cur_attack_name_list
    mutation_rate_list = config.fine_mutation_rate_list
    detect_method_list = ["Precision_mean","Precision_var","Loss_mean","Loss_var","Recall_mean","Recall_var",
                        "Entropy_model_mean","Entropy_model_var","Entropy_sample_mean","Entropy_sample_var",
                        "LCR_model_mean","LCR_sample_var"
                        ]
    for dataset_name in dataset_name_list:
        data[dataset_name] = {}
        if dataset_name == "CIFAR10":
            class_num = 10
        if dataset_name == "GTSRB":
            class_num = 43
        for model_name in model_name_list:
            data[dataset_name][model_name] = {}
            for attack_name in attack_name_list:
                dataset_name[dataset_name][model_name][attack_name] = {}
                for mutated_rate in mutation_rate_list:
                    dataset_name[dataset_name][model_name][attack_name][mutated_rate] = {}
                    # 预测标签df
                    df_Label = load_df(dataset_name,model_name,attack_name,mutated_rate,"preLabel")
                    # 选择出top50变异模型
                    mutated_model_global_id_list = get_top_k_global_ids(df_Label,top_k=50,trend="bigger")
                    df_CELoss = load_df(dataset_name,model_name,attack_name,mutated_rate,"CELoss")
                    for detect_method in detect_method_list:
                        class_rank,target_class_ranking_percent = detect_method_pool(df_Label,df_CELoss,class_num,mutated_model_global_id_list,detect_method)
                        dataset_name[dataset_name][model_name][attack_name][mutated_rate][detect_method] = {
                            "class_rank":class_rank,
                            "target_class_ranking_percent":target_class_ranking_percent
                        }
    for mutated_rate in mutation_rate_list:
        
        for detect_method in detect_method_list:
            for attack_name in attack_name_list:
                for dataset_name in dataset_name_list:
                    for model_name in model_name_list:



if __name__ == "__main__":
    main()
    



