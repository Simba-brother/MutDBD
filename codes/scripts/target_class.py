import sys
sys.path.append("./")
import os
import joblib
import statistics
import numpy as np
from scipy import stats
from codes import config
from cliffs_delta import cliffs_delta

dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_name_list = config.mutation_name_list
mutation_rate_list = config.mutation_rate_list
exp_root_dir = config.exp_root_dir

class_num = 10
mutated_model_num = 50
mutated_operator_num = len(mutation_name_list)

def get_target_class_step_1(mutation_name):
    ans_dic = {}
    for mutation_rate in mutation_rate_list:
        ans_dic[mutation_rate] = {}
        for class_idx in range(class_num):
            ans_dic[mutation_rate][class_idx] = []
    eval_poisoned_trainset_report_path = os.path.join(exp_root_dir,dataset_name,model_name,attack_name,mutation_name, "eval_poisoned_trainset_report.data")
    eval_poisoned_trainset_report = joblib.load(eval_poisoned_trainset_report_path)
    for mutation_rate in mutation_rate_list:
        report_list = eval_poisoned_trainset_report[mutation_rate]
        for report in report_list:
            for class_idx in range(class_num):
                precision = report[str(class_idx)]["precision"]
                ans_dic[mutation_rate][class_idx].append(precision)
    assert len(ans_dic[0.01][0]) == 50, "数量不对"
    return ans_dic

def get_target_class_step_2(data_dic):
    ans_dic = {}
    for mutation_rate in mutation_rate_list:
        ans_dic[mutation_rate] = {"max_mean_value_class_idx":-1, "max_median_value_class_idx":-1, "target_class_idx":-1}

    for mutation_rate in mutation_rate_list:
        max_mean_value = 0
        max_mean_value_class_idx = -1
        max_median_value = 0
        max_median_value_class_idx = -1
        for class_idx in range(class_num):
            mean_value = np.mean(data_dic[mutation_rate][class_idx])
            median_value = np.median(data_dic[mutation_rate][class_idx])
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_value_class_idx = class_idx
            if median_value > max_median_value:
                max_median_value = median_value
                max_median_value_class_idx = class_idx
        ans_dic[mutation_rate]["max_mean_value_class_idx"] = max_mean_value_class_idx
        ans_dic[mutation_rate]["max_median_value_class_idx"] = max_median_value_class_idx
        if max_mean_value_class_idx == max_median_value_class_idx:
            ans_dic[mutation_rate]["target_class_idx"] = max_mean_value_class_idx
    return ans_dic




def get_target_class(mutation_name):
    ans_dict_1 = get_target_class_step_1(mutation_name)
    ans_dict_2 = get_target_class_step_2(ans_dict_1)
    print(ans_dict_2)


def get_target_class_all_mutator():
    ans_dict_1 = get_target_class_all_mutation_step_1()
    ans_dict_2 = get_target_class_step_2(ans_dict_1)
    print(ans_dict_2)

def get_adaptive_ratio_step_1(data_dict_1, data_dict_2):
    res = {}
    
    for mutation_rate in mutation_rate_list:
        target_class_idx = data_dict_2[mutation_rate]["target_class_idx"]
        if target_class_idx == -1:
            continue
        source_list = data_dict_1[mutation_rate][target_class_idx]
        other_list_list = []
        for class_idx in range(class_num):
            if class_idx == target_class_idx:
                continue
            other_list_list.append(data_dict_1[mutation_rate][class_idx])
        p_value_list = []
        cliff_delta_list = []
        for other_list in other_list_list:
            if source_list == other_list:
                p_value_list.append(float("inf"))
                cliff_delta_list.append(0.0)
                continue
            # 计算p值
            p_value = stats.wilcoxon(source_list, other_list).pvalue
            p_value_list.append(p_value)
            # 计算clif delta
            source_list_sorted = sorted(source_list)
            other_list_sorted = sorted(other_list)
            d,info = cliffs_delta(source_list_sorted, other_list_sorted)
            cliff_delta_list.append(abs(d))
        res[mutation_rate] = {
            "target_class_i":target_class_idx,
            "p_value_list":p_value_list,
            "clif_delta_list":cliff_delta_list
        }
    return res

def get_adaptive_ratio_step_2(data_dict):
    res = -1
    target_class_i = -1
    candidate_mutation_ratio_list = sorted(list(data_dict.keys()))
    min_sum_p_value = float("inf")
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

def get_adaptive_ratio():
    ans = {}
    for cur_mutation_name in mutation_name_list:
        ans_dict_1 = get_target_class_step_1(cur_mutation_name)
        ans_dict_2 = get_target_class_step_2(ans_dict_1)
        ans_dict_3 = get_adaptive_ratio_step_1(ans_dict_1, ans_dict_2)
        adaptive_rate, target_class_i = get_adaptive_ratio_step_2(ans_dict_3)
        ans[cur_mutation_name] = {"adaptive_rate":adaptive_rate,  "target_class_i":target_class_i}
    return ans

def get_target_class_all_mutation_step_1():
    ans_dic = {}
    for mutation_rate in mutation_rate_list:
        ans_dic[mutation_rate] = {}
        for class_idx in range(class_num):
            ans_dic[mutation_rate][class_idx] = []

    for cur_mutation_name in mutation_name_list:
        temp_dict = get_target_class_step_1(cur_mutation_name)
        for cur_mutation_rate in mutation_rate_list:
            for class_idx in range(class_num):
                precision_list = temp_dict[cur_mutation_rate][class_idx]
                ans_dic[cur_mutation_rate][class_idx].extend(precision_list)
    assert len(ans_dic[0.01][0]) == mutated_model_num*mutated_operator_num, "数量不对"
    return ans_dic

def get_target_class_adptive_rate():
    ans = {}
    adaptive_rate_dict = get_adaptive_ratio()
    for cur_mutation_name in mutation_name_list:
        adaptive_rate = adaptive_rate_dict[cur_mutation_name]["adaptive_rate"]
        ans_dict_1 = get_target_class_step_1(mutation_name)
        ans_dict_2 = get_target_class_step_2(ans_dict_1)
        target_class_idx = ans_dict_2[adaptive_rate]["target_class_idx"]
        ans[cur_mutation_name] = target_class_idx
    return ans

if __name__ == "__main__":
    # get_target_class_all_mutator()
    get_adaptive_ratio()
    pass
