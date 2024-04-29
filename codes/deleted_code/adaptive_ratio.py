import sys
from collections import defaultdict
import torch
import os
import joblib
sys.path.append("./")
from codes.eval_model import EvalModel
from codes.utils import create_dir
from codes import config
from codes import draw
from scipy import stats
import numpy as np
import math
from cliffs_delta import cliffs_delta

dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_name_list =  config.mutation_name_list
attack_name_list = config.attack_name_list

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
backdoor_model = dict_state["backdoor_model"]    
testset = dict_state["poisoned_trainset"]
device = torch.device("cuda:0")
mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
mutation_model_num = 50
exp_root_dir = "/data/mml/backdoor_detect/experiments"

e = EvalModel(backdoor_model, testset, device)
origin_report = e._eval_classes_acc()

def get_mutation_ratio_class_dif_dic_singel(dataset_name, model_name, attack_name, mutation_name):
    '''
    res[0.01][0]: 0.01变异率下,类别0的dif list
    '''
    file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    file_path = os.path.join(exp_root_dir,file_name)
    data = joblib.load(file_path)
    res = {}
    for mutation_ratio in mutation_ratio_list:
        res[mutation_ratio] = defaultdict(list)
        for m_i in range(mutation_model_num):
            for class_i in range(10):
                cur_precision = data[mutation_ratio][m_i][str(class_i)]["precision"]
                origin_precision = origin_report[str(class_i)]["precision"]
                dif = origin_precision - cur_precision
                res[mutation_ratio][class_i].append(dif)
    return res

def get_mutation_ratio_class_dif_dic_all():
    # 存各个变异方法
    data_list = []
    # 遍历变异方法
    for mutation_name in  config.mutation_name_list:
        file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
        file_path = os.path.join("/data/mml/backdoor_detect/experiments",file_name)
        data = joblib.load(file_path)
        data_list.append(data)

    res = {}
    for mutation_ratio in mutation_ratio_list:
        res[mutation_ratio] = {}
        for class_i in range(10):
            res[mutation_ratio][class_i] = []
    for data in data_list:
        for mutation_ratio in mutation_ratio_list:
            report_list = data[mutation_ratio]
            for report in report_list:
                for class_i in range(10):
                    dif = origin_report[str(class_i)]["precision"] - report[str(class_i)]["precision"]
                    res[mutation_ratio][class_i].append(dif)
    return res

def get_mutation_ratio_min_class(data_dic):
    '''
    res[0.01] = {
        "min_mean_class_i",1
        "min_median_class_i",1
    }
    解释: 0.01变异率下均值最小类和中位值最小类
    '''
    res = defaultdict()
    for mutation_ratio in mutation_ratio_list:
        min_mean_class_i = -1
        min_median_class_i = -1
        min_mean = float('inf')
        min_median = float("inf")
        for class_i in range(10):
            dif_list = data_dic[mutation_ratio][class_i]
            mean = np.mean(dif_list)
            median = np.median(dif_list)     
            if mean < min_mean:
                min_mean = mean
                min_mean_class_i = class_i
            if median < min_median:
                min_median = median
                min_median_class_i = class_i
        res[mutation_ratio] = {"min_mean_class_i":min_mean_class_i, "min_median_class_i": min_median_class_i}
    return res

def get_mutation_ratio_target_class(data_dic):
    '''
    res[0.01]: 0.01变异率下,target class若为-1表示根据最小均值和最小中位值确定不出来target class
    '''
    res = {}
    for mutation_ratio in mutation_ratio_list:
        min_mean_class_i = data_dic[mutation_ratio]["min_mean_class_i"]
        min_median_class_i = data_dic[mutation_ratio]["min_median_class_i"]
        if min_mean_class_i == min_median_class_i:
            res[mutation_ratio] = min_mean_class_i
        else:
            res[mutation_ratio] = -1
    return res

def get_mutation_ratio_target_class_pValue_clif_list(data_dic_1, data_dic_2):
    '''
    res[0.01] = {
        "target_class_i": 1
        "p_value_list":[_,_,]
    }
    解释: 0.01变异率的target class和其与other target class的p_value
    '''
    res = {}
    for mutation_ratio in mutation_ratio_list:
        target_class_i = data_dic_2[mutation_ratio]
        if target_class_i == -1:
            continue
        target_dif_list = data_dic_1[mutation_ratio][target_class_i]
        cliff_delta_list = []
        p_value_list = []
        for class_i in range(10):
            if class_i == target_class_i:
                continue
            dif_list = data_dic_1[mutation_ratio][class_i]
            if dif_list == target_dif_list:
                p_value_list.append(float("inf"))
                cliff_delta_list.append(0.0)
            else:
                wil = stats.wilcoxon(target_dif_list, dif_list)
                p_value_list.append(wil.pvalue)
                target_list_sorted = sorted(target_dif_list)
                list_sorted = sorted(dif_list)
                d,info = cliffs_delta(target_list_sorted, list_sorted)
                cliff_delta_list.append(abs(d))
        res[mutation_ratio] = {
            "target_class_i":target_class_i,
            "p_value_list":p_value_list,
            "clif_delta_list":cliff_delta_list
        }
    return res
    
def get_candidate_ratio(data_dict):
    res  = {
            "candidate_ratio_1":[],
            "candidate_ratio_2":[],
            "candidate_ratio_3":None,
            }
    min_sum = float('inf')
    for mutation_ratio in data_dict.keys():
        target_class_i = data_dict[mutation_ratio]["target_class_i"]
        p_value_list = data_dict[mutation_ratio]["p_value_list"]
        mean_p_value = np.mean(p_value_list)
        cur_sum = sum(p_value_list)

        all_P_flag = all(p_value < 0.05 for p_value in p_value_list)
        if all_P_flag is True:
            res["candidate_ratio_1"].append((mutation_ratio,cur_sum))
        if mean_p_value < 0.05:
            res["candidate_ratio_2"].append((mutation_ratio,cur_sum))
        if cur_sum < min_sum:
            min_sum = cur_sum
            res["candidate_ratio_3"] = mutation_ratio

    temp_list = res["candidate_ratio_1"]
    temp_list = sorted(temp_list, key=lambda x: x[1])
    res["candidate_ratio_1"] = []
    for x in temp_list:
        res["candidate_ratio_1"].append(x[0])
    temp_list = res["candidate_ratio_2"]
    temp_list = sorted(temp_list, key=lambda x: x[1])
    res["candidate_ratio_2"] = []
    for x in temp_list:
        res["candidate_ratio_2"].append(x[0])
    return res

def get_candidate_ratio_2(data_dict):
    res = -1
    target_class_i = -1
    candidate_mutation_ratio_list = sorted(list(data_dict.keys()))
    for mutation_ratio in candidate_mutation_ratio_list:
        p_value_list = data_dict[mutation_ratio]["p_value_list"]
        clif_delta_list = data_dict[mutation_ratio]["clif_delta_list"]
        all_P_flag = all(p_value < 0.05 for p_value in p_value_list)
        all_C_flag = all(d >= 0.147 for d in clif_delta_list)
        if all_P_flag is True and all_C_flag is True:
            res = mutation_ratio
            target_class_i = data_dict[mutation_ratio]["target_class_i"]
            break
    return res, target_class_i
    
def adaptive_single():
    '''
    return: adaptive_ratio_dic
        adaptive_ratio_dic[attack_name][mutation_name] = {"adaptive_ratio":candidate_ratio, "target_class_idx":target_class_i}
    '''
    adaptive_ratio_dic = {}
    for attack_name in attack_name_list:
        adaptive_ratio_dic[attack_name] = {}
        for mutation_name in mutation_name_list:
            res_1 = get_mutation_ratio_class_dif_dic_singel(dataset_name, model_name, attack_name, mutation_name)
            res_2 = get_mutation_ratio_min_class(res_1)
            res_3 = get_mutation_ratio_target_class(res_2)
            res_4 = get_mutation_ratio_target_class_pValue_clif_list(res_1, res_3)
            candidate_ratio, target_class_i = get_candidate_ratio_2(res_4)
            adaptive_ratio_dic[attack_name][mutation_name] = {"adaptive_ratio":candidate_ratio, "target_class_idx":target_class_i}
    print(adaptive_ratio_dic)
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name)
    create_dir(save_dir)
    save_file_name = "adaptive_ratio_dic.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    joblib.dump(adaptive_ratio_dic, save_file_path)
    return adaptive_ratio_dic

def adaptive_all_2(adaptive_ratio_dic):
    res = {}
    for attack_name in attack_name_list:
        ans = defaultdict(list)
        for mutation_name in mutation_name_list:
            dif_dic = get_mutation_ratio_class_dif_dic_singel(dataset_name, model_name, attack_name, mutation_name)
            adaptive_ratio = adaptive_ratio_dic[attack_name][mutation_name]["adaptive_ratio"]
            if adaptive_ratio == -1:
                # 跳过当前攻击下的这个变异方式
                continue
            for class_idx in range(10):
                ans[class_idx].extend(dif_dic[adaptive_ratio][class_idx])
        min_mean_class_i = -1
        min_median_class_i = -1
        min_mean = float('inf')
        min_median = float("inf")
        for class_idx in range(10):
            mean = np.mean(ans[class_idx])
            median = np.median(ans[class_idx])
            if mean < min_mean:
                min_mean = mean
                min_mean_class_i = class_idx
            if median < min_median:
                min_median = median
                min_median_class_i = class_idx
        if min_mean_class_i == min_median_class_i:
            target_class = min_mean_class_i
        else:
            target_class = -1
        res[attack_name] = target_class
    print(res)

def adaptive_all():
    res_1 = get_mutation_ratio_class_dif_dic_all()
    res_2 = get_mutation_ratio_min_class(res_1)
    res_3 = get_mutation_ratio_target_class(res_2)
    res_4 = (res_1, res_3)
    candidate_ratio = get_candidate_ratio(res_4)
    print(candidate_ratio)
   

def draw_box(adaptive_ratio_dic):
    # 数据集/模型/攻击名称
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    origin_report = e._eval_classes_acc()
    # 存各个变异方法
    data_list = []
    # 遍历变异方法
    for mutation_name in  config.mutation_name_list:
        file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
        file_path = os.path.join("/data/mml/backdoor_detect/experiments",file_name)
        data = joblib.load(file_path)
        data_list.append((mutation_name,data))
    ans = defaultdict(list)
    mutated_model_num = len(data_list)*50
    for data in data_list:
        mutation_name =data[0]
        adaptive_ratio = adaptive_ratio_dic[attack_name][mutation_name]["adaptive_ratio"]
        if adaptive_ratio == -1:
            mutated_model_num -= 50
            continue
        report_list = data[1][adaptive_ratio]
        for report in report_list:
            for class_i in range(10):
                dif = origin_report[str(class_i)]["precision"] - report[str(class_i)]["precision"]
                ans[class_i].append(dif)

    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box", dataset_name, model_name)
    create_dir(save_dir)
    all_y = []
    labels = []
    for class_i in range(10):
        y_list = ans[class_i]
        all_y.append(y_list)
        labels.append(f"Class_{class_i}")
    title = f"{dataset_name}_{model_name}_{attack_name}_adptive_merge"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    draw.draw_box(all_y, labels,title,save_path)
    print(f"mutated_model_num:{mutated_model_num}")


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
    
def draw_box_of_adaptive_ratio(adaptive_ratio_dic):
    # 数据集/模型/攻击名称
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    # original backdoor model在各个类别上的precision和accuracy
    origin_report = e._eval_classes_acc()
    # 存储 数据集/模型/攻击下，[(变异算子，{变异率:[report]}),...]
    data_list = []
    # 遍历变异方法
    for mutation_name in  config.mutation_name_list:
        report_path = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_name, "eval_poisoned_trainset_report.data")
        data = joblib.load(report_path)
        data_list.append((mutation_name,data))
    # 存储{class_idx:[precision_o-precision_m]}
    ans = defaultdict(list)
    mutated_model_num = len(data_list)*50
    # 遍历变异算子
    for data in data_list:
        # 得到变异算子名称
        mutation_name =data[0]
        # 得到该变异算子下的自适应变异率 攻击:变异算子:自适应变异率
        adaptive_ratio = adaptive_ratio_dic[attack_name][mutation_name]["adaptive_ratio"]
        if adaptive_ratio == -1:
            mutated_model_num -= 50
            continue
        report_list = data[1][adaptive_ratio]
        for report in report_list:
            for class_i in range(10):
                dif = origin_report[str(class_i)]["precision"] - report[str(class_i)]["precision"]
                ans[class_i].append(dif)
    
    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box", dataset_name, model_name, attack_name, "adptive_ratio")
    create_dir(save_dir)
    all_y = []
    labels = []
    for class_i in range(10):
        y_list = ans[class_i]
        all_y.append(y_list)
        labels.append(f"Class_{class_i}")
    title = f"{dataset_name}_{model_name}_{attack_name}_adptive_ratio"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "Category"
    ylabel = "Precision difference"
    draw.draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{mutated_model_num}")

def draw_box_of_target_class_adaptive_ratio(adaptive_ratio_dic):
    # 数据集/模型/攻击名称
    backdoor_model = dict_state["backdoor_model"]
    # 纯污染集
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    # 纯clean集
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    # 目标class
    target_class_idx = 1
    # 目标class中的污染集
    poisoned_set_target_class = TargetClassDataset(purePoisonedTrainDataset, target_class_idx)
    # 目标class中的干净集
    clean_set_target_class = TargetClassDataset(pureCleanTrainDataset, target_class_idx)
    # 目标class中的污染集的accuracy
    e = EvalModel(backdoor_model, poisoned_set_target_class, device)
    poisoned_origin_acc = e._eval_acc()
    # 目标class中的干净集的accuracy
    e = EvalModel(backdoor_model, clean_set_target_class, device)
    clean_origin_acc = e._eval_acc()
    # 存储 数据集/模型/攻击下，[(变异算子，{变异率:[{"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole}]}),...]
    data_list = []
    # 遍历变异方法
    for mutation_name in  config.mutation_name_list:
        report_path = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_name, "eval_poisoned_trainset_target_class.data")
        data = joblib.load(report_path)
        data_list.append((mutation_name,data))
    # 存储{"clean":[accuracy_o-accuracy_m], "poisoned":[accuracy_o-accuracy_m]}
    ans = defaultdict(list)
    mutated_model_num = len(data_list)*50
    # 遍历变异算子
    for data in data_list:
        # 得到变异算子名称
        mutation_name =data[0]
        # 得到该变异算子下的自适应变异率 攻击:变异算子:自适应变异率
        adaptive_ratio = adaptive_ratio_dic[attack_name][mutation_name]["adaptive_ratio"]
        if adaptive_ratio == -1:
            mutated_model_num -= 50
            continue
        dic_list = data[1][adaptive_ratio]
        for dic in dic_list:
            clean_acc = dic["target_class_clean_acc"]
            poisoned_acc = dic["target_class_poisoned_acc"]
            dif_clean = clean_origin_acc - clean_acc
            dif_poisoned = poisoned_origin_acc - poisoned_acc
            ans["target_class_clean"].append(dif_clean)
            ans["target_class_poisoned"].append(dif_poisoned)
    
    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box", dataset_name, model_name, attack_name, "adptive_ratio", "targetClass")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(ans["target_class_clean"])
    all_y.append(ans["target_class_poisoned"])
    labels.append("target_class_clean")
    labels.append("target_class_poisoned")
    
    title = f"{dataset_name}_{model_name}_{attack_name}_adptive_ratio"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "Group"
    ylabel = "Accuracy difference"
    draw.draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{mutated_model_num}")


if __name__ == "__main__":
    # adaptive_ratio_dic = adaptive_single()

    adapive_ratio_dic_path = os.path.join(exp_root_dir, dataset_name, model_name, "adaptive_ratio_dic.data")
    adaptive_ratio_dic = joblib.load(adapive_ratio_dic_path)
    # draw_box_of_adaptive_ratio(adaptive_ratio_dic)
    draw_box_of_target_class_adaptive_ratio(adaptive_ratio_dic)
    # adaptive_all_2(adaptive_ratio_dic)
    # adaptive_all()