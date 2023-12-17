import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import joblib
sys.path.append("./")
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes import draw
from codes.utils import create_dir
from codes import config
from scipy import stats
import numpy as np

dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_name = config.mutation_name 
mutation_name_list =  config.mutation_name_list
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


dict_state = get_dict_state()
backdoor_model = dict_state["backdoor_model"]    
testset = dict_state["poisoned_trainset"]
device = torch.device("cuda:7")
mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
mutation_model_num = 50
exp_root_dir = "/data/mml/backdoor_detect/experiments"

e = EvalModel(backdoor_model, testset, device)
origin_report = e._eval_classes_acc()

def get_mutation_ratio_class_dif_dic_singel():
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
    res = {}
    for mutation_ratio in mutation_ratio_list:
        min_mean_class_i = data_dic[mutation_ratio]["min_mean_class_i"]
        min_median_class_i = data_dic[mutation_ratio]["min_median_class_i"]
        if min_mean_class_i == min_median_class_i:
            res[mutation_ratio] = min_mean_class_i
        else:
            res[mutation_ratio] = -1
    return res

def get_mutation_ratio_target_class_p_value_list(data_dic_1, data_dic_2):
    res = {}
    for mutation_ratio in mutation_ratio_list:
        target_class_i = data_dic_2[mutation_ratio]
        if target_class_i == -1:
            continue
        target_dif_list = data_dic_1[mutation_ratio][target_class_i]
        p_value_list = []
        for class_i in range(10):
            if class_i == target_class_i:
                continue
            dif_list = data_dic_1[mutation_ratio][class_i]
            if dif_list == target_dif_list:
                p_value_list.append(float("inf"))
            else:
                wil = stats.wilcoxon(target_dif_list, dif_list)
                p_value_list.append(wil.pvalue)
        res[mutation_ratio] = {
            "target_class_i":target_class_i,
            "p_value_list":p_value_list
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

def adaptive_single():
    res_1 = get_mutation_ratio_class_dif_dic_singel()
    res_2 = get_mutation_ratio_min_class(res_1)
    res_3 = get_mutation_ratio_target_class(res_2)
    res_4 = get_mutation_ratio_target_class_p_value_list(res_1, res_3)
    candidate_ratio = get_candidate_ratio(res_4)
    print(candidate_ratio)


def adaptive_all():
    res_1 = get_mutation_ratio_class_dif_dic_all()
    res_2 = get_mutation_ratio_min_class(res_1)
    res_3 = get_mutation_ratio_target_class(res_2)
    res_4 = get_mutation_ratio_target_class_p_value_list(res_1, res_3)
    candidate_ratio = get_candidate_ratio(res_4)
    print(candidate_ratio)
   
if __name__ == "__main__":
    adaptive_single()