import sys
sys.path.append("./")
import numpy as np
import random
from collections import defaultdict
import os
import statistics
import joblib
import setproctitle
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import scipy.stats as stats
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from codes.datasets.cifar10.models.resnet18_32_32_3 import ResNet
from codes.datasets.cifar10.models.vgg import VGG
from codes.eval_model import EvalModel
from codes.utils import entropy, create_dir
from codes.scripts.baseData import BaseData
import scipy.stats as stats
import queue
from codes import config
from codes.draw import draw_box,draw_line,draw_stackbar
from cliffs_delta import cliffs_delta
from mutated_model_selected import select_by_suspected_nonsuspected_acc_dif, select_by_suspected_nonsuspected_confidence_distribution_dif

# 随机数种子
random.seed(555)
# 模型结构
model = ResNet(18)
# model = VGG("VGG19")
# 攻击name
attack_name = "IAD" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet
# 实验结果文件夹
exp_root_dir = "/data/mml/backdoor_detect/experiments"
# 数据集name
dataset_name = "CIFAR10"
# model name
model_name = "resnet18_nopretrain_32_32_3" # resnet18_nopretrain_32_32_3, vgg19
# 变异模型存储文件夹path
mutates_path = os.path.join(exp_root_dir, dataset_name, model_name, "mutates")
# 变异算子name
mutation_name_list = ["gf","neuron_activation_inverse","neuron_block","neuron_switch","weight_shuffle"]
# dataset/model/各个攻击方式，各个变异算子，对应的自适应变异率
adapive_ratio_dic_path = os.path.join(exp_root_dir, dataset_name, model_name, "adaptive_ratio_dic.data")
adaptive_ratio_dic = joblib.load(adapive_ratio_dic_path)

if dataset_name == "CIFAR10":
    if model_name == "resnet18_nopretrain_32_32_3":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import *
    if model_name == "vgg19":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_vgg19 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_vgg19 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_vgg19 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_vgg19 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_vgg19 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_vgg19 import *

dict_state = get_dict_state()
# attack category
target_class_idx = 1
# 攻击类别数据集
class TargetClassDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx  = target_class_idx
        self.target_class_dataset = self.get_target_class_dataset()

    def get_target_class_dataset(self):
        target_class_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label == self.target_class_idx:
                target_class_dataset.append((sample, label))
        return target_class_dataset
    
    def __len__(self):
        return len(self.target_class_dataset)
    
    def __getitem__(self, index):
        x,y=self.target_class_dataset[index]
        return x,y
# 非攻击类别数据集
class NoTargetClassDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx  = target_class_idx
        self.notarget_class_dataset = self.get_target_class_dataset()

    def get_target_class_dataset(self):
        notarget_class_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label != self.target_class_idx:
                notarget_class_dataset.append((sample, label))
        return notarget_class_dataset
    
    def __len__(self):
        return len(self.notarget_class_dataset)
    
    def __getitem__(self, index):
        x,y=self.notarget_class_dataset[index]
        return x,y
# 非怀疑数据集
class NoSuspectedClassDataset(Dataset):
    def __init__(self, dataset, suspected_class_idx_list):
        self.dataset = dataset
        self.suspected_class_idx_list  = suspected_class_idx_list
        self.nosuspected_dataset = self._get_nosuspected_dataset()

    def _get_nosuspected_dataset(self):
        nosuspected_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label not in self.suspected_class_idx_list:
                nosuspected_dataset.append((sample, label))
        return nosuspected_dataset
    
    def __len__(self):
        return len(self.nosuspected_dataset)
    
    def __getitem__(self, index):
        x,y=self.nosuspected_dataset[index]
        return x,y
# 怀疑集
class SuspectedClassDataset(Dataset):
    def __init__(self, dataset, suspected_class_idx_list):
        self.dataset = dataset
        self.suspected_class_idx_list  = suspected_class_idx_list
        self.suspected_dataset = self._get_suspected_dataset()

    def _get_suspected_dataset(self):
        suspected_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label in self.suspected_class_idx_list:
                suspected_dataset.append((sample, label))
        return suspected_dataset
    
    def __len__(self):
        return len(self.suspected_dataset)
    
    def __getitem__(self, index):
        x,y=self.suspected_dataset[index]
        return x,y
# 后门模型结构
backdoor_model = dict_state["backdoor_model"]
pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]

poisoned_trainset = dict_state["poisoned_trainset"]
target_class_dataset_clean = TargetClassDataset(pureCleanTrainDataset, target_class_idx)
target_class_dataset_poisoned = TargetClassDataset(purePoisonedTrainDataset, target_class_idx)
no_target_class_trainset = NoTargetClassDataset(poisoned_trainset, target_class_idx)

device = torch.device("cuda:0")

def get_mutation_models_weight_file_path(mutation_ratio):
    weight_filePath_list = []
    for mutation_name in mutation_name_list:
        mutation_name_dir_path = os.path.join(mutates_path, mutation_name)
        if mutation_name  == "gf":
            ratio_dir = f"ratio_{mutation_ratio}_scale_5_num_50"
        else:
            ratio_dir = f"ratio_{mutation_ratio}_num_50"
        mutated_models_weight_dir_path = os.path.join(mutation_name_dir_path, ratio_dir, attack_name) 
        for m_i in range(50):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"model_mutated_{m_i+1}.pth")
            weight_filePath_list.append(weight_path)
    assert len(weight_filePath_list) == len(mutation_name_list) * 50, "出错了,总的变异数目应该是250"
    print("get_mutation_models_weight_file_path() successfully")
    return weight_filePath_list

def get_mutation_models_weight_file_path_2(adaptive_ratio_dic):
    weight_filePath_list = []
    for mutation_name in mutation_name_list:
        mutation_ratio = adaptive_ratio_dic[attack_name][mutation_name]["adaptive_ratio"]
        if mutation_ratio == -1:
            continue
        mutation_name_dir_path = os.path.join(mutates_path, mutation_name)
        if mutation_name  == "gf":
            ratio_dir = f"ratio_{mutation_ratio}_scale_5_num_50"
        else:
            ratio_dir = f"ratio_{mutation_ratio}_num_50"
        mutated_models_weight_dir_path = os.path.join(mutation_name_dir_path, ratio_dir, attack_name) 
        for m_i in range(50):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"model_mutated_{m_i+1}.pth")
            weight_filePath_list.append(weight_path)
    assert len(weight_filePath_list) >=  50, "出错了,总的变异数目应该是250"
    print("get_mutation_models_weight_file_path() successfully")
    return weight_filePath_list

def remove_models(weight_filePath_list, dataset):
    reserved_weight_filePath_list = []
    e = EvalModel(backdoor_model, dataset, device)
    acc_o = e._eval_acc()
    for m_i, weight_filePath in tqdm(enumerate(weight_filePath_list)):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        acc_m = e._eval_acc()
        if acc_m == acc_o or acc_m > acc_o:
            continue
        else:
            reserved_weight_filePath_list.append(weight_filePath)
    return reserved_weight_filePath_list

def split_models(weight_filePath_list, dataset):
    acc_list = []
    e = EvalModel(backdoor_model, dataset, device)
    for m_i, weight_filePath in tqdm(enumerate(weight_filePath_list)):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        acc = e._eval_acc()
        acc_list.append(acc)
    sorted_tuples = sorted(enumerate(acc_list), key=lambda x: x[1])
    indices = [i for i, _ in sorted_tuples]
    sorted_w_list = []
    for i in indices:
        sorted_w_list.append(weight_filePath_list[i])
    
    cut_off_1 = int(len(sorted_w_list)*0.25)
    cut_off_2 = int(len(sorted_w_list)*0.75)
    return sorted_w_list[cut_off_1:cut_off_2]

def get_pred_label(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in tqdm(enumerate(weight_filePath_list)):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        pred_labels = e._get_pred_labels()
        res[f"m_{m_i}"] = pred_labels
    df = pd.DataFrame(res)
    print("get_pred_label() successfully")
    return df

def get_confidence(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in tqdm(enumerate(weight_filePath_list)):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        confidence_list = e._get_confidence_list()
        res[f"m_{m_i}"] = confidence_list
    df = pd.DataFrame(res)
    print("get_confidence() successfully")
    return df

def get_trueOrFalse(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in tqdm(enumerate(weight_filePath_list)):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        trueOrFalse_list = e._get_TrueOrFalse()
        res[f"m_{m_i}"] = trueOrFalse_list
    df = pd.DataFrame(res)
    print("get_pred_label() successfully")
    return df

def get_output(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        outputs = e._get_outputs()
        res[f"m_{m_i}"] = outputs
    print("get_output() successfully")
    return res

def get_mean_entropy(df:pd.DataFrame):
    entropy_list = []
    for row_id, row in df.iterrows():
        entropy_list.append(entropy(list(row)))
    average = statistics.mean(entropy_list)
    print("get_mean_entropy() successfully")
    return average

def get_entropy_list(df:pd.DataFrame):
    entropy_list = []
    for row_id, row in df.iterrows():
        entropy_list.append(entropy(list(row)))
    print("get_entropy_list() successfully")
    return entropy_list

def get_confidence_list(df:pd.DataFrame):
    confidence_list = []
    for row_id, row in df.iterrows():
        confidence_list.append(np.mean(list(row)))
    print("get_confidence_list() successfully")
    return confidence_list

def get_true_count_list(df:pd.DataFrame):
    true_count_list = []
    for row_id, row in df.iterrows():
        true_count_list.append(sum(list(row)))
    print("get_true_count_list() successfully")
    return true_count_list

# 根据样本数据和置信水平求置信区间
def confidence_interval(data, alpha=0.01):
    '''
    alpha:0.01 # 置信水平为99%
    '''
    n = len(data)  # 样本数
    mean = np.mean(data)  # 样本均值
    std = np.std(data)  # 样本标准差
    t_value = stats.t.ppf(1 - alpha / 2, df=n-1)  # t分布临界值
    margin_error = t_value * std / (n ** 0.5)  # 边际误差
    lower_limit = mean - margin_error  # 置信下限
    upper_limit = mean + margin_error  # 置信上限
    return (lower_limit, upper_limit)

def sort_model_by_acc(weight_filePath_list:list, dataset):
    '''
    acc越低,模型优先级越高
    '''
    sorted_weight_filePath_list = []
    acc_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        acc = e._eval_acc()
        acc_list.append(acc)
    sorted_model_indices = sorted(range(len(acc_list)), key=lambda x: acc_list[x])
    for i in sorted_model_indices:
        sorted_weight_filePath_list.append(weight_filePath_list[i])
    return sorted_weight_filePath_list

def calu_deepGini(output):
    cur_sum = 0
    for v in output:
        cur_sum += v*v
    return 1-cur_sum

def sort_model_by_deepGini(weight_filePath_list:list, dataset):
    '''
    deepgini越大,优先级越高
    '''
    sorted_weight_filePath_list = []
    deepGini_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        outputs = e._get_outputs()
        avg_deepGini = 0
        for output in outputs:
            avg_deepGini += calu_deepGini(output)
        avg_deepGini = avg_deepGini / len(outputs)
        deepGini_list.append(avg_deepGini)
    sorted_model_indices = sorted(range(len(deepGini_list)), key=lambda x: deepGini_list[x])
    sorted_model_indices.reverse()
    for i in sorted_model_indices:
        sorted_weight_filePath_list.append(weight_filePath_list[i])
    return sorted_weight_filePath_list

def select_top_k_models(sorted_weight_filePath_list, k=50):
    cut_off = 0.5
    cut_point = int(len(sorted_weight_filePath_list)*cut_off)
    temp_list = sorted_weight_filePath_list[:cut_point]
    ans = random.sample(temp_list, k)
    return ans

def save_df(df, save_dir, save_file_name):
    create_dir(save_dir)
    save_path = os.path.join(save_dir, save_file_name)
    df.to_csv(save_path, index=False)

    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path(mutation_ratio=0.05)
    print("变异模型个数:", len(weight_filePath_list))
    # 得到预测标签
    df = get_pred_label(weight_filePath_list, target_class_dataset_poisoned)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_poisoned_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    poisoned_average_entropy = get_mean_entropy(df)
    print("poisoned_average_entropy:", poisoned_average_entropy)


    df = get_pred_label(weight_filePath_list, target_class_dataset_clean)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_clean_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    clean_average_entropy = get_mean_entropy(df)
    print("clean_average_entropy:", clean_average_entropy)

    print("==="*10)
    print("clean_average_entropy:", clean_average_entropy)
    print("poisoned_average_entropy:", poisoned_average_entropy)

def get_center_and_distance_list(df:pd.DataFrame):
    means = df.mean()
    center = means.tolist()
    distance_list = []
    for row_idx, row in df.iterrows():
        point = row.tolist()
        distance = 0
        for x,y in zip(point, center):
            distance += abs(x-y)
        distance_list.append(distance)
    return center, distance_list

def calculate_distance(point_1, point_2):
    distance = 0
    for x, y in zip(point_1, point_2):
        distance+=abs(x-y)
    return distance

def get_suspected_class_idx_list(adaptive_ratio_dic):
    '''
    args:
        adaptive_ratio_dic: 
            adaptive_ratio_dic[attack_name][mutation_name][target_class_idx]:获得该攻击下变异方式预测的目标class
    return:
        获得某个攻击下,所有变异方式的target_class作为suspected_class_idx_list
    '''
    suspected_class_idx_list = []
    for mutation_name in mutation_name_list:
        suspected_class_idx_list.append(adaptive_ratio_dic[attack_name][mutation_name]["target_class_idx"])

    return list(set(suspected_class_idx_list))

def get_acc_list(weight_filePath_list, nosuspected_dataset):
    acc_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, nosuspected_dataset, device)
        acc = e._eval_acc()
        acc_list.append(acc)
    return acc_list

def fliter_mutated_model(weight_filePath_list, suspicious_dataset, non_suspicious_dataset):
    filter_w_list = []
    cliff_delta_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, suspicious_dataset, device=device)
        suspicious_confidence_list = e._get_confidence()
        mean_suspicious = np.mean(suspicious_confidence_list)
        median_suspicious = np.median(suspicious_confidence_list)
        e = EvalModel(model, non_suspicious_dataset,device=device)
        non_suspicious_confidence_list = e._get_confidence()
        mean_no_suspicious = np.mean(non_suspicious_confidence_list)
        median_no_suspicious = np.median(non_suspicious_confidence_list)
        # if mean_suspicious > mean_no_suspicious and median_suspicious > median_no_suspicious:
        # wil = stats.wilcoxon(suspicious_confidence_list, non_suspicious_confidence_list)
        # p_value_list.append(wil.pvalue)
        sorted_suspicious_confidence_list = sorted(suspicious_confidence_list)
        sorted_non_suspicious_confidence_list = sorted(non_suspicious_confidence_list)
        d,info = cliffs_delta(sorted_suspicious_confidence_list, sorted_non_suspicious_confidence_list)
        cliff_delta_list.append(abs(d))
        all_data = [suspicious_confidence_list, non_suspicious_confidence_list]
        labels = ["suspicious", "non_suspicious"]
        title = f"model_{m_i}_confidence"
        save_dir = os.path.join(exp_root_dir,f"images/box/{dataset_name}/{model_name}/select_mutated_model")
        save_file_name = f"model_{m_i}.png"
        save_path = os.path.join(save_dir, save_file_name)
        draw_box(all_data, labels, title, save_path)
    negative_cliff_delta_list = [-x for x in cliff_delta_list]
    sorted_w_i_list = np.argsort(negative_cliff_delta_list)
    top_w_i_list = sorted_w_i_list[:50]
    for i in top_w_i_list:
        filter_w_list.append(weight_filePath_list[i]) 
    return filter_w_list

def priorityQueue_2_list(q:queue.PriorityQueue):
    qsize = q.qsize()
    res = []
    while not q.empty():
        res.append(q.get())
    assert len(res) == qsize, "队列数量不对"
    return res

def detect_by_suspected_nonsuspected_acc_dif_top_model_truecount():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))


    sample_global_idx = 0
    # 怀疑子集中的真实clean的
    q = queue.PriorityQueue()

    df_clean_suspected = get_trueOrFalse(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        sample_global_idx += 1
        priority = sum(row.tolist()) # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        ground_truth_label = False # 真实标记为clean
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (-priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample
    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_trueOrFalse(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        sample_global_idx += 1
        priority = sum(row.tolist()) # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        ground_truth_label = True # 真实标记为clean
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (-priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample)

    priority_list = priorityQueue_2_list(q)
    if attack_name != "IAD":
        assert len(priority_list) == len(suspected_dataset), "数量不对"
    # 截取前 50% 
    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    for cut_off in cut_off_list:
        end = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:end]
        TP = 0
        FP = 0
        gt_TP = len(suspected_poisoned_dataset)
        for item in prefix_priority_list:
            gt_label = item[3]
            if gt_label == True:
                TP += 1
            else:
                FP += 1
        precision = round(TP/(TP+FP),3)
        recall = round(TP/gt_TP,3)
        precision_list.append(precision)
        recall_list.append(recall)
        print("FP:",FP)
        print("TP:",TP)
        print("precision:",precision)
        print("recall:",recall)
        print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
        print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))
    y = {"precision":precision_list, "recall":recall_list}
    title = "The relationship between detection performance and cut off"
    save_dir = os.path.join(exp_root_dir, "images", "line", dataset_name, model_name, "acc_dif_truecount")
    create_dir(save_dir)
    save_filename  = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_filename)
    draw_line(cut_off_list, title, save_path, **y)

def get_label_maintenance_ratio_list(df:pd.DataFrame):
    res = []
    for row_idx, row in df.iterrows():
        res.append(sum(row.tolist())/len(row.tolist()))  
    return res

def detect_by_suspected_nonsuspected_acc_dif_top_model_truecount_interval():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    # no-suspected dataset
    df_no_suspected = get_trueOrFalse(weight_filePath_list, nosuspected_dataset)
    label_maintenance_ratio_list = get_label_maintenance_ratio_list(df_no_suspected)
    interval = stats.t.interval(confidence=0.99, df=len(label_maintenance_ratio_list)-1, loc=np.mean(label_maintenance_ratio_list), scale=stats.sem(label_maintenance_ratio_list))

    pred_list = []
    gt_list = []
    # 怀疑子集中的真实clean的
    df_clean_suspected = get_pred_label(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        cur_maintenance_ratio = sum(row.tolist())/len(row.tolist())   # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        if cur_maintenance_ratio > interval[1]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(False) 

    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_pred_label(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        cur_maintenance_ratio = sum(row.tolist())/len(row.tolist())   # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        if cur_maintenance_ratio > interval[1]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(True) 

    if attack_name != "IAD":
        assert len(pred_list) == len(suspected_dataset), "数量不对"
        assert len(gt_list) == len(suspected_dataset), "数量不对"

    TN, FP, FN, TP = confusion_matrix(gt_list, pred_list).ravel()
    acc = round((TP+TN)/(TP+TN+FP+FN),3)
    precision = round(TP/(TP+FP),3)
    recall = round(TP/(TP+FN),3)
    F1 = round(2*precision*recall/(precision+recall),3)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)
    print("TP:",TP)
    print("acc:",acc)
    print("precision:",precision)
    print("recall:",recall)
    print("F1:",F1)
    print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
    print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))

def detect_by_suspected_nonsuspected_acc_dif_top_model_entropy():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))


    sample_global_idx = 0
    q = queue.PriorityQueue()
    # 怀疑子集中的真实clean的
    df_clean_suspected = get_pred_label(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        sample_global_idx += 1
        priority = entropy(row.tolist()) # 值越小，说明样本越鲁棒，也就是越有可能为backdoor
        ground_truth_label = False # 真实标记为clean
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample
    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_pred_label(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        sample_global_idx += 1
        # 值越小，说明样本越鲁棒，也就是越有可能为backdoor
        priority = entropy(row.tolist()) 
        # 真实标记为clean
        ground_truth_label = True 
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample)
    priority_list = priorityQueue_2_list(q)
    if attack_name != "IAD":
        assert len(priority_list) == len(suspected_dataset), "数量不对"
    # 截取前 50% 
    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    for cut_off in cut_off_list:
        end = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:end]
        TP = 0
        FP = 0
        gt_TP = len(suspected_poisoned_dataset)
        for item in prefix_priority_list:
            gt_label = item[3]
            if gt_label == True:
                TP += 1
            else:
                FP += 1
        precision = round(TP/(TP+FP),3)
        recall = round(TP/gt_TP,3)
        precision_list.append(precision)
        recall_list.append(recall)
        print("FP:",FP)
        print("TP:",TP)
        print("precision:",precision)
        print("recall:",recall)
        print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
        print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))
    y = {"precision":precision_list, "recall":recall_list}
    title = "The relationship between detection performance and cut off"
    save_dir = os.path.join(exp_root_dir, "images", "line", dataset_name, model_name, "acc_dif_entropy")
    create_dir(save_dir)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_file_name)
    draw_line(cut_off_list, title, save_path, **y)

def detect_by_suspected_nonsuspected_acc_dif_top_model_entropy_interval():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    # nosuspected dataset
    df_nosuspected = get_pred_label(weight_filePath_list, nosuspected_dataset)
    entropy_list = get_entropy_list(df_nosuspected)
    interval = stats.t.interval(confidence=0.99, df=len(entropy_list)-1, loc=np.mean(entropy_list), scale=stats.sem(entropy_list))

    pred_list = []
    gt_list = []
    # 怀疑子集中的真实clean的
    df_clean_suspected = get_pred_label(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        cur_e = entropy(row.tolist())  # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        if cur_e < interval[0]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(False) 

    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_pred_label(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        cur_e = entropy(row.tolist())  # 值越大，说明样本越鲁棒，也就是越有可能为backdoor
        if cur_e < interval[0]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(True) 

    if attack_name != "IAD":
        assert len(pred_list) == len(suspected_dataset), "数量不对"
        assert len(gt_list) == len(suspected_dataset), "数量不对"

    TN, FP, FN, TP = confusion_matrix(gt_list, pred_list).ravel()
    acc = round((TP+TN)/(TP+TN+FP+FN),3)
    precision = round(TP/(TP+FP),3)
    recall = round(TP/(TP+FN),3)
    F1 = round(2*precision*recall/(precision+recall),3)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)
    print("TP:",TP)
    print("acc:",acc)
    print("precision:",precision)
    print("recall:",recall)
    print("F1:",F1)
    print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
    print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))

def detect_by_suspected_nonsuspected_confidence_distribution_dif_center_distance():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_confidence_distribution_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    df_nosuspected = get_confidence(weight_filePath_list, nosuspected_dataset)
    center, distance_list = get_center_and_distance_list(df_nosuspected)


    sample_global_idx = 0
    q = queue.PriorityQueue()

    # 怀疑子集中的真实clean的
    df_clean_suspected = get_confidence(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        sample_global_idx += 1
        cur_point = row.tolist()
        cur_distance = 0
        for x,y in zip(cur_point, center):
            cur_distance += abs(x-y)
        priority = cur_distance # 值越大，越有可能为backdoor
        ground_truth_label = False # 真实标记为clean
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (-priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample
    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_confidence(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        sample_global_idx += 1
        cur_point = row.tolist()
        cur_distance = 0
        for x,y in zip(cur_point, center):
            cur_distance += abs(x-y)
        priority = cur_distance # 值越大，越有可能为backdoor
        ground_truth_label = True # 真实标记为clean
        # 0:优先级, 1:样本global_id, 2: 样本local_id, 3:gt_label
        item = (-priority, sample_global_idx, row_idx,  ground_truth_label) 
        q.put(item) # 队列越靠前越有可能为backdoor sample
    priority_list = priorityQueue_2_list(q)
    if attack_name != "IAD":
        assert len(priority_list) == len(suspected_dataset), "数量不对"
    
    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    for cut_off in cut_off_list:
        end = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:end]
        TP = 0
        FP = 0
        gt_TP = len(suspected_poisoned_dataset)
        for item in prefix_priority_list:
            gt_label = item[3]
            if gt_label == True:
                TP += 1
            else:
                FP += 1
        precision = round(TP/(TP+FP),3)
        recall = round(TP/gt_TP,3)
        precision_list.append(precision)
        recall_list.append(recall)
        print("FP:",FP)
        print("TP:",TP)
        print("precision:",precision)
        print("recall:",recall)
        print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
        print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))
    y = {"precision":precision_list, "recall":recall_list}
    title = "The relationship between detection performance and cut off"
    save_dir = os.path.join(exp_root_dir, "images", "line", dataset_name, model_name, "confidence_distribution_dif_center_distance")
    create_dir(save_dir)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_file_name)
    draw_line(cut_off_list, title, save_path, **y)

def detect_by_suspected_nonsuspected_confidence_distribution_dif_center_distance_interval():
    '''
    dataset/attack/model
    clean和poisoned检测
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 获得没有被怀疑的class的dataset，可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的dataset，将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 获得被怀疑的class的真实clean dataset
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 获得被怀疑的class的真实poisoned dataset
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_confidence_distribution_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    # nosuspected dataset
    df_nosuspected = get_confidence(weight_filePath_list, nosuspected_dataset)
    center, distance_list = get_center_and_distance_list(df_nosuspected)
    interval = stats.t.interval(confidence=0.99, df=len(distance_list)-1, loc=np.mean(distance_list), scale=stats.sem(distance_list))

    pred_list = []
    gt_list = []
    # 怀疑子集中的真实clean的
    df_clean_suspected = get_confidence(weight_filePath_list, suspected_clean_dataset)
    for row_idx, row in df_clean_suspected.iterrows():
        cur_point = row.tolist()
        cur_distance = 0
        for x,y in zip(cur_point, center):
            cur_distance += abs(x-y)
        if cur_distance > interval[1]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(False) 

    # 怀疑子集中的真实poisoned的
    df_poisoned_suspected = get_confidence(weight_filePath_list, suspected_poisoned_dataset)
    for row_idx, row in df_poisoned_suspected.iterrows():
        cur_point = row.tolist()
        cur_distance = 0
        for x,y in zip(cur_point, center):
            cur_distance += abs(x-y)
        if cur_distance > interval[1]:
            pred_list.append(True)
        else:
            pred_list.append(False)
        gt_list.append(True) 

    if attack_name != "IAD":
        assert len(pred_list) == len(suspected_dataset), "数量不对"
        assert len(gt_list) == len(suspected_dataset), "数量不对"

    TN, FP, FN, TP = confusion_matrix(gt_list, pred_list).ravel()
    acc = round((TP+TN)/(TP+TN+FP+FN),3)
    precision = round(TP/(TP+FP),3)
    recall = round(TP/(TP+FN),3)
    F1 = round(2*precision*recall/(precision+recall),3)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)
    print("TP:",TP)
    print("acc:",acc)
    print("precision:",precision)
    print("recall:",recall)
    print("F1:",F1)
    print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
    print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))

def vote_truecount():
    '''
    统计并绘制,在变异model上,样本满足条件
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 非怀疑集,可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集,将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集中的真实clean
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 怀疑集中的真实poisoned
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio权重
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    df_clean_suspected = get_trueOrFalse(weight_filePath_list, suspected_clean_dataset)
    df_poisoned_suspected = get_trueOrFalse(weight_filePath_list, suspected_poisoned_dataset)
    clean_count_list = []
    poisoned_count_list = []
    guan_ka_num_list = list(reversed([x for x in range(0, 51)])) # [50,49,...,1,0]
    for guan_ka_num in guan_ka_num_list:
        clean_count = 0
        poisoned_count = 0
        for row_idx, row in df_clean_suspected.iterrows():
            truecount = sum(row.tolist())
            if truecount == guan_ka_num:
                clean_count += 1
        clean_count_list.append(clean_count)
        for row_idx, row in df_poisoned_suspected.iterrows():
            truecount = sum(row.tolist())
            if truecount == guan_ka_num:
                poisoned_count += 1
        poisoned_count_list.append(poisoned_count)
    save_dir = os.path.join(exp_root_dir, "images", "bar", dataset_name, model_name, "acc_dif", "TrueCount")
    create_dir(save_dir)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_file_name)
    x_ticks = [str(x) for x in guan_ka_num_list]
    draw_stackbar(x_ticks =x_ticks, title="The relationship between the number of clean samples and the number of poisoned samples in the conditional model", save_path=save_path, y_1_list=clean_count_list, y_2_list=poisoned_count_list)

def vote_difcount():
    '''
    统计并绘制,在变异model上,样本满足条件
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 非怀疑集,可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集,将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集中的真实clean
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 怀疑集中的真实poisoned
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio权重
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_acc_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    df_clean_suspected = get_trueOrFalse(weight_filePath_list, suspected_clean_dataset)
    df_poisoned_suspected = get_trueOrFalse(weight_filePath_list, suspected_poisoned_dataset)
    clean_count_list = []
    poisoned_count_list = []
    guan_ka_num_list = [x for x in range(-50, 51)] # [-50,-49,...,-1,0,1,2,...,49,50]
    for guan_ka_num in guan_ka_num_list:
        clean_count = 0
        poisoned_count = 0
        for row_idx, row in df_clean_suspected.iterrows():
            truecount = sum(row.tolist())
            falsecount = len(row)-truecount
            dif = truecount - falsecount
            if dif == guan_ka_num:
                clean_count += 1
        clean_count_list.append(clean_count)
        for row_idx, row in df_poisoned_suspected.iterrows():
            truecount = sum(row.tolist())
            falsecount = len(row)-truecount
            dif = truecount - falsecount
            if dif == guan_ka_num:
                poisoned_count += 1
        poisoned_count_list.append(poisoned_count)
    save_dir = os.path.join(exp_root_dir, "images", "bar", dataset_name, model_name, "acc_dif", "count_dif")
    create_dir(save_dir)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_file_name)
    x_ticks = [str(x) for x in guan_ka_num_list]
    draw_stackbar(x_ticks =x_ticks, title="The relationship between the number of clean samples and the number of poisoned samples in the conditional model", save_path=save_path, y_1_list=clean_count_list, y_2_list=poisoned_count_list)

def vote_confidence():
    '''
    统计并绘制,在变异model上,样本满足条件
    '''
    # 获得attack name下被变异模型怀疑的target class idx list
    suspected_class_idx_list = get_suspected_class_idx_list(adaptive_ratio_dic)
    if attack_name == "IAD":
        assert len(pureCleanTrainDataset)+2*len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    else:
        assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"
    # 非怀疑集,可作为我们的clean samples
    nosuspected_dataset = NoSuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集,将用于检测
    suspected_dataset = SuspectedClassDataset(poisoned_trainset, suspected_class_idx_list)
    # 怀疑集中的真实clean
    suspected_clean_dataset = SuspectedClassDataset(pureCleanTrainDataset, suspected_class_idx_list)
    # 怀疑集中的真实poisoned
    suspected_poisoned_dataset = SuspectedClassDataset(purePoisonedTrainDataset, suspected_class_idx_list)
    # 加载adaptive_ratio权重
    weight_filePath_list = get_mutation_models_weight_file_path_2(adaptive_ratio_dic)
    print("变异模型个数:", len(weight_filePath_list))
    weight_filePath_list = select_by_suspected_nonsuspected_confidence_distribution_dif(model, weight_filePath_list, suspected_dataset, nosuspected_dataset, device)
    print("剩余变异模型个数:", len(weight_filePath_list))
    df_no_suspected = get_confidence(weight_filePath_list, nosuspected_dataset)
    center, distance_list = get_center_and_distance_list(df_no_suspected)
    df_clean_suspected = get_confidence(weight_filePath_list, suspected_clean_dataset)
    df_poisoned_suspected = get_confidence(weight_filePath_list, suspected_poisoned_dataset)
    clean_count_list = []
    poisoned_count_list = []
    guan_ka_num_list = list(reversed([x for x in range(0, 51)])) # [50,49,...,1,0]
    for guan_ka_num in guan_ka_num_list:
        clean_count = 0
        poisoned_count = 0
        for row_idx, row in df_clean_suspected.iterrows():
            vote_count = 0
            confidence_list = row.tolist()
            for x1,x2 in zip(confidence_list, center):
                if x1 > x2:
                    vote_count+=1
            if vote_count == guan_ka_num:
                clean_count += 1
        clean_count_list.append(clean_count)
        for row_idx, row in df_poisoned_suspected.iterrows():
            vote_count = 0
            confidence_list = row.tolist()
            for x1,x2 in zip(confidence_list, center):
                if x1 > x2:
                    vote_count+=1
            if vote_count == guan_ka_num:
                poisoned_count += 1
        poisoned_count_list.append(poisoned_count)
    save_dir = os.path.join(exp_root_dir, "images", "bar", dataset_name, model_name, "acc_dif", "confidence_vote")
    create_dir(save_dir)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir, save_file_name)
    x_ticks = [str(x) for x in guan_ka_num_list]
    draw_stackbar(x_ticks =x_ticks, title="The relationship between the number of clean samples and the number of poisoned samples in the conditional model", save_path=save_path, y_1_list=clean_count_list, y_2_list=poisoned_count_list)



def get_targetclass_clean_poisoned_accuracy_variation():
    mutation_operator_name = "gf"
    bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
    mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    mean_p_acc_list = []
    mean_c_acc_list = []
    for mutation_rate in mutation_rate_list:
        poisoned_acc_list = []
        clean_acc_list = []
        mutation_weight_file_list = bdata._get_mutation_weight_file(mutation_rate)
        for mutation_weight_file in mutation_weight_file_list:
            model.load_state_dict(torch.load(mutation_weight_file, map_location="cpu"))
            e = EvalModel(model, target_class_dataset_poisoned, device) 
            p_acc = e._eval_acc()
            e = EvalModel(model, target_class_dataset_clean, device) 
            c_acc = e._eval_acc()
            poisoned_acc_list.append(p_acc)
            clean_acc_list.append(c_acc)
        mean_p_acc = sum(poisoned_acc_list) / len(poisoned_acc_list)
        mean_c_acc = sum(clean_acc_list) / len(clean_acc_list)
        mean_p_acc_list.append(mean_p_acc)
        mean_c_acc_list.append(mean_c_acc)
    # 画图
    x_ticks = mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = "targetclass_clean_poisoned_accuracy_variation"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"poisoned set":mean_p_acc_list, "clean set":mean_c_acc_list}
    draw_line(x_ticks, title, xlabel, save_path, **y)
    print("get_targetclass_clean_poisoned_accuracy_variation() successful")

def get_class_precision_variation_by_mutation_operator_name(mutation_operator_name = "gf"):
    bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
    report_data = bdata.get_eval_poisoned_trainset_report_data()
    mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

    temp_dict = defaultdict(dict)
    for class_i in range(10):
        for mutation_rate in mutation_rate_list:
            temp_dict[class_i][mutation_rate] = []

    for mutation_rate in mutation_rate_list:
        report_list = report_data[mutation_rate]
        for report in report_list:
            for class_i in range(10):
                precision = report[str(class_i)]["precision"]
                temp_dict[class_i][mutation_rate].append(precision)

    y_dict = defaultdict(list)
    for class_i in range(10):
        for mutation_rate in mutation_rate_list:
           y_dict[f"class_{class_i}"].append(np.mean(temp_dict[class_i][mutation_rate]))
    x_ticks =  mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = "class_precision_variation.png"
    save_path = os.path.join(save_dir, save_file_name) 
    draw_line(x_ticks, title, xlabel, save_path, **y_dict)
    print("get_class_precision_variation() successful")

def get_class_precision_variation_by_all_mutation_operator_name():
    mutation_operator_name_list = config.mutation_name_list
    report_dict = {}
    for mutation_operator_name in mutation_operator_name_list:
        bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
        report_data = bdata.get_eval_poisoned_trainset_report_data()
        report_dict[mutation_operator_name] = report_data
    mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    temp_dict = defaultdict(dict)
    for class_i in range(10):
        for mutation_rate in mutation_rate_list:
            temp_dict[class_i][mutation_rate] = []
    for mutation_rate in mutation_rate_list:
        for mutation_operator_name in mutation_operator_name_list:
            report_data = report_dict[mutation_operator_name]
            report_list = report_data[mutation_rate]
            for report in report_list:
                for class_i in range(10):
                    precision = report[str(class_i)]["precision"]
                    temp_dict[class_i][mutation_rate].append(precision)
    y_dict = defaultdict(list)
    for class_i in range(10):
        for mutation_rate in mutation_rate_list:
           y_dict[f"class_{class_i}"].append(np.mean(temp_dict[class_i][mutation_rate]))
    x_ticks =  mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:All"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, "All")
    create_dir(save_dir)
    save_file_name = "class_precision_variation.png"
    save_path = os.path.join(save_dir, save_file_name) 
    draw_line(x_ticks, title, xlabel, save_path, **y_dict)
    print("get_class_precision_variation() successful")

def get_target_class_poisoned_clean_dif_variation_by_mutation_operator_name(mutation_operator_name = "gf"):
    bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
    mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    report_data = bdata.get_eval_poisoned_trainset_target_class_report_data()
    y_dict = defaultdict(list)
    for mutation_rate in mutation_rate_list:
        report_list = report_data[mutation_rate]
        for report in report_list:
            dif = report["target_class_poisoned_acc"] - report["target_class_clean_acc"]
            y_dict[mutation_rate].append(dif)
    all_y = []
    labels = []
    for mutation_rate in mutation_rate_list:
        all_y.append(y_dict[mutation_rate])
        labels.append(str(mutation_rate))
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
    xlabel = "mutation_rate"
    ylabel = "Acc(poisoned) - Acc(clean)"
    save_dir = os.path.join(exp_root_dir,"images/box", dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = "target_class_poisoned_clean_dif_variation.png"
    save_path = os.path.join(save_dir, save_file_name) 
    draw_box(data=all_y, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)
    print("get_target_class_poisoned_clean_dif_variation_by_mutation_operator_name() success")


def get_target_class_poisoned_clean_dif_variation_by_All_mutation_operator():
    mutation_operator_name_list = config.mutation_name_list
    report_dict = {}
    for mutation_operator_name in mutation_operator_name_list:
        bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
        report_data = bdata.get_eval_poisoned_trainset_target_class_report_data()
        report_dict[mutation_operator_name] = report_data
    
    y_dict = defaultdict(list)
    mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    for mutation_rate in mutation_rate_list:
        for mutation_operator_name in mutation_operator_name_list:
            report_data = report_dict[mutation_operator_name]
            report_list = report_data[mutation_rate]
            for report in report_list:
                dif = report["target_class_poisoned_acc"] - report["target_class_clean_acc"]
                y_dict[mutation_rate].append(dif)
    all_y = []
    labels = []
    for mutation_rate in mutation_rate_list:
        all_y.append(y_dict[mutation_rate])
        labels.append(str(mutation_rate))
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:All"
    xlabel = "mutation_rate"
    ylabel = "Acc(poisoned) - Acc(clean)"
    save_dir = os.path.join(exp_root_dir,"images/box", dataset_name, model_name, attack_name, "All")
    create_dir(save_dir)
    save_file_name = "target_class_poisoned_clean_dif_variation.png"
    save_path = os.path.join(save_dir, save_file_name) 
    draw_box(data=all_y, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)
    print("get_target_class_poisoned_clean_dif_variation_by_mutation_operator_name() success")

def get_target_class_poisoned_clean_dif_variation_by_All_mutation_operator_adaptive_rate():
    mutation_operator_name_list = config.mutation_name_list

    mutation_operator_dict = {}
    for mutation_operator_name in mutation_operator_name_list:
        adaptive_ratio = adaptive_ratio_dic[attack_name][mutation_operator_name]["adaptive_ratio"]
        mutation_operator_dict[mutation_operator_name] = adaptive_ratio

    report_dict = {}
    for mutation_operator_name in mutation_operator_name_list:
        bdata = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
        report_data = bdata.get_eval_poisoned_trainset_target_class_report_data()
        report_dict[mutation_operator_name] = report_data

    y_dict = defaultdict(list)
    for mutation_operator_name in mutation_operator_name_list:
        report_data = report_dict[mutation_operator_name]
        mutation_rate = mutation_operator_dict[mutation_operator_name]
        if mutation_rate == -1:
            continue
        report_list = report_data[mutation_rate]
        for report in report_list:
            dif = report["target_class_poisoned_acc"] - report["target_class_clean_acc"]
            y_dict["adaptive_mutation_rate"].append(dif)
    all_y = []
    labels = []
    all_y = y_dict["adaptive_mutation_rate"]
    labels.append("adaptive_mutation_rate")
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, adaptive_rate:{mutation_operator_dict}"
    xlabel = "adaptive_mutation_rate"
    ylabel = "Acc(poisoned) - Acc(clean)"
    save_dir = os.path.join(exp_root_dir,"images/box", dataset_name, model_name, attack_name, "All_adaptive_rate")
    create_dir(save_dir)
    save_file_name = "target_class_poisoned_clean_dif_variation.png"
    save_path = os.path.join(save_dir, save_file_name) 
    draw_box(data=all_y, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)
    print("get_target_class_poisoned_clean_dif_variation_by_All_mutation_operator_adaptive_rate() success")
    

if __name__ == "__main__":
    # 模型选择:suspected_nonsuspected_confidence_distribution_dif 
    ## 根据变异模型预测confidence
    # detect_by_suspected_nonsuspected_confidence_distribution_dif_center_distance()
    # detect_by_suspected_nonsuspected_confidence_distribution_dif_center_distance_interval()

    # 模型选择: suspected_nonsuspected_acc_dif 
    ## 根据变异模型预测标签维持率
    # detect_by_suspected_nonsuspected_acc_dif_top_model_truecount()
    # detect_by_suspected_nonsuspected_acc_dif_top_model_truecount_interval()
    ## 根据变异模型预测标签熵
    # detect_by_suspected_nonsuspected_acc_dif_top_model_entropy()
    # detect_by_suspected_nonsuspected_acc_dif_top_model_entropy_interval()
    # vote_truecount()
    # vote_difcount()
    # vote_confidence()
    # setproctitle.setproctitle(attack_name)
    # get_targetclass_clean_poisoned_accuracy_variation()

    # get_class_precision_variation_by_mutation_operator_name()
    # get_class_precision_variation_by_all_mutation_operator_name()

    # get_target_class_poisoned_clean_dif_variation_by_All_mutation_operator()
    get_target_class_poisoned_clean_dif_variation_by_All_mutation_operator_adaptive_rate()
    pass
