import sys
sys.path.append("./")
import os
import joblib
from collections import defaultdict
import numpy as np
from scipy import stats
from cliffs_delta import cliffs_delta
from codes.draw import draw_box
from codes.utils import create_dir
from codes import config


dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_name_list = config.mutation_name_list
mutation_rate_list = config.mutation_rate_list
from codes.scripts.target_class import get_adaptive_rate_of_Hybrid_mutator, get_adaptive_ratio_of_Combin_mutator
exp_root_dir = config.exp_root_dir
class_num = config.class_num

mutated_model_num = 50
mutated_operator_num = len(mutation_name_list)


def get_clean_poisoned_in_target_class_of_hybrid_mutator_accuracy_list():
    '''
    所有变异算子混合 > 在不同变异率下 > 变异模型们在target_class中clean和poisoned的accuracy list
    '''
    # 返回的数据
    data_dic = {}
    # 初始化数据结构
    # {mutation_rate:{"clean":[], "poisoned":[]}}
    for mutation_rate in mutation_rate_list:
        data_dic[mutation_rate] = {"clean":[], "poisoned":[]}
    for mutation_name in mutation_name_list:
        eval_poisoned_trainset_target_class_path = os.path.join(exp_root_dir,dataset_name,model_name,attack_name, mutation_name, "eval_poisoned_trainset_target_class.data")
        eval_poisoned_trainset_target_class_report = joblib.load(eval_poisoned_trainset_target_class_path)
        for mutation_rate in mutation_rate_list:
            report_list = eval_poisoned_trainset_target_class_report[mutation_rate]
            for report in report_list:
                clean_acc = report["target_class_clean_acc"]
                poisoned_acc = report["target_class_poisoned_acc"]
                data_dic[mutation_rate]["clean"].append(clean_acc)
                data_dic[mutation_rate]["poisoned"].append(poisoned_acc)
    assert len(data_dic[0.01]["clean"]) == 50*mutated_operator_num, "数量不对"

    
    return data_dic

def analysis_clean_poisoned_in_target_class_of_Hybrid_mutator_with_adaptive_rate():
    is_dif = False
    higher = None
    dic_1,_ = get_adaptive_rate_of_Hybrid_mutator()
    adaptive_rate = dic_1["adaptive_rate"]
    dic_2 = get_clean_poisoned_in_target_class_of_hybrid_mutator_accuracy_list()
    clean_acc_list = dic_2[adaptive_rate]["clean"]
    poisoned_acc_list = dic_2[adaptive_rate]["poisoned"]
    # 计算pvalue
    p_value = stats.wilcoxon(clean_acc_list, poisoned_acc_list).pvalue
    if p_value < 0.05:
        is_dif = True
    clean_acc_list_sorted = sorted(clean_acc_list)
    poisoned_acc_list_sorted = sorted(poisoned_acc_list)
    d,info = cliffs_delta(clean_acc_list_sorted, poisoned_acc_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"

    # 绘图：
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Hybrid_clean_poisoned", "adaptive_rate")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_acc_list)
    all_y.append(poisoned_acc_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Hybrid_adaptive_rate_{adaptive_rate}"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Accuracy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{mutated_model_num*mutated_operator_num}")
    print("analysis_clean_poisoned_in_target_class() success")
    return is_dif, higher


def get_clean_poisoned_in_target_class_with_dif_mutation_rate_by_mutation_name(mutation_name):
    '''
    得到某个变异算子下 > 在不同变异率下 > 变异模型们在各个类别上的precision
    '''
    # 返回的数据
    ans_dic = {}
    # 数据结构初始化
    # {mutation_rate:{class_idx:[precision]}}
    for mutation_rate in mutation_rate_list:
        ans_dic[mutation_rate] = {"clean":[], "poisoned":[]}
    eval_poisoned_trainset_target_class_path = os.path.join(exp_root_dir,dataset_name,model_name,attack_name, mutation_name, "eval_poisoned_trainset_target_class.data")
    eval_poisoned_trainset_target_class_report = joblib.load(eval_poisoned_trainset_target_class_path)
    for mutation_rate in mutation_rate_list:
        report_list = eval_poisoned_trainset_target_class_report[mutation_rate]
        for report in report_list:
            clean_acc = report["target_class_clean_acc"]
            poisoned_acc = report["target_class_poisoned_acc"]
            ans_dic[mutation_rate]["clean"].append(clean_acc)
            ans_dic[mutation_rate]["poisoned"].append(poisoned_acc)
    assert len(ans_dic[0.01]["clean"]) == mutated_model_num, "数量不对"
    return ans_dic

def get_clean_poisoned_acc_list_in_target_class_of_combin_mutator():
    dic = get_adaptive_ratio_of_Combin_mutator()
    clean_acc_list = []
    poisoned_acc_list = []
    for mutation_name in config.mutation_name_list:
        adaptive_rate = dic[mutation_name]["adaptive_rate"]
        if adaptive_rate == -1:
            continue
        dic_1 = get_clean_poisoned_in_target_class_with_dif_mutation_rate_by_mutation_name(mutation_name)
        temp_clean_acc_list = dic_1[adaptive_rate]["clean"]
        temp_poisoned_acc_list = dic_1[adaptive_rate]["poisoned"]
        clean_acc_list.extend(temp_clean_acc_list)
        poisoned_acc_list.extend(temp_poisoned_acc_list)
    print("cur_mutated_model_num:",len(clean_acc_list)) 
    return clean_acc_list, poisoned_acc_list

def analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate():
    is_dif = False
    higher = None
    clean_acc_list, poisoned_acc_list = get_clean_poisoned_acc_list_in_target_class_of_combin_mutator()
    # 计算pvalue
    if clean_acc_list == poisoned_acc_list:
        p_value = float("inf")
    else:
        p_value = stats.wilcoxon(clean_acc_list, poisoned_acc_list).pvalue
    if p_value < 0.05:
        is_dif = True
    clean_acc_list_sorted = sorted(clean_acc_list)
    poisoned_acc_list_sorted = sorted(poisoned_acc_list)
    d,info = cliffs_delta(clean_acc_list_sorted, poisoned_acc_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"
    # 绘图：
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Combin_clean_poisoned", "adaptive_rate")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_acc_list)
    all_y.append(poisoned_acc_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Combin_adaptive_rate"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Accuracy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{len(clean_acc_list)}")
    print("analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate() success")
    return is_dif, higher




if __name__ == "__main__":

    is_dif, higher = analysis_clean_poisoned_in_target_class_of_Hybrid_mutator_with_adaptive_rate()
    print("==================")
    print(f"dataset:{dataset_name}")
    print(f"model:{model_name}")
    print(f"attack:{attack_name}")
    print(f"is_dif:{is_dif}")
    print(f"higher:{higher}")


    # is_dif, higher = analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate()
    # print("==================")
    # print(f"dataset:{dataset_name}")
    # print(f"model:{model_name}")
    # print(f"attack:{attack_name}")
    # print(f"is_dif:{is_dif}")
    # print(f"higher:{higher}")