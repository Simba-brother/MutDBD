# 用于计算class rank
import os
import joblib
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from utils.small_utils import nested_defaultdict
from utils.calcu_utils import entropy
import numpy as np
import time
from utils.dataset_utils import get_class_num
from defense.our.mutation.mutation_select import get_top_k_global_ids
from utils.common_utils import convert_to_hms
from utils.common_utils import get_formattedDateTime
from collections import defaultdict

def one_scene(dataset_name, model_name, attack_name, mutation_rate=0.01, metric="FP", save_dir=None):
    '''获得class rank list and rank top'''
    df_predicted_labels = pd.read_csv(os.path.join(exp_root_dir,"EvalMutationToCSV",dataset_name,model_name,attack_name,str(mutation_rate),"preLabel.csv"))
    mutated_model_id_list = get_top_k_global_ids(df_predicted_labels,top_k=50,trend="bigger")
    class_num = get_class_num(dataset_name)
    if metric == "FP":
        res = FP_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    elif metric == "Precision":
        res = precision_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    elif metric == "F1":
        res = f1_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    elif metric == "Recall":
        res = recall_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    elif metric == "LCR":
        res = LCR_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    elif metric == "Entropy":
        res = entropy_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    else:
        raise ValueError("Invalid input")
    if save_dir:
        # 保存数据
        save_file_name = f"{metric}.joblib"
        save_path = os.path.join(save_dir,save_file_name)
        joblib.dump(res,save_path)
        print("保存:",save_path)
    record = {
        "metric":metric,
        "data_save_dir":save_dir,
        "rank_ratio":res["target_class_rank_ratio"]
    }
    return record

def main_batch_sciences(dataset_name_list, model_name_list, attack_name_list):
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                print(f"{dataset_name}|{model_name}|{attack_name}")
                one_scene(dataset_name, model_name, attack_name, metric="FP")
    
def recall_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num:int):
    '''从小到大'''
    # {class_id(int):[recall_1,recall_2,..recall_500]}
    data_dict = defaultdict(list)
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(class_num):
            recall = report[str(class_i)]["recall"]
            data_dict[class_i].append(recall)
    # {class_id(int):avg_recall}
    dict_2 = {}
    for class_id in range(class_num):
        dict_2[class_id] = round(np.mean(data_dict[class_id]),4)
    sorted_res_1 = dict(sorted(dict_2.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def FP_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num):
    '''从大到小'''
    data_stuct_1 = nested_defaultdict(2,int)
    for m_i in mutated_model_global_id_list:
        pre_labels = df[f"model_{m_i}"]
        gt_labels = df[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(class_num):
            data_stuct_1[m_i][class_i] = np.sum(cm[:,class_i]) - cm[class_i][class_i]

    data_stuct_2 = nested_defaultdict(1,int)
    for class_i in range(class_num):
        for m_i in mutated_model_global_id_list:
            data_stuct_2[class_i] += data_stuct_1[m_i][class_i]
    sorted_res_1 = dict(sorted(data_stuct_2.items(), key=lambda x: x[1],reverse=True))
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def f1_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num:int):
    '''从小到大'''
    # {class_id(int):[f1_1,f1_2,..f1_500]}
    data_dict = defaultdict(list)
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(class_num):
            f1 = report[str(class_i)]["f1-score"]
            data_dict[class_i].append(f1)
    # {class_id(int):avg_f1}
    dict_2 = {}
    for class_id in range(class_num):
        dict_2[class_id] = round(np.mean(data_dict[class_id]),4)
    sorted_res_1 = dict(sorted(dict_2.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def LCR_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num:int):
    '''从小到大'''
    # {class_id(int):avg_LCR}
    data_dict = defaultdict(list)
    for class_id in range(class_num):
        # 抽取出某个类别的df
        class_df = df.loc[df["GT_label"]==class_id]
        # 遍历该类别所有的样本
        lcr_samples = []
        for row_idx,row in class_df.iterrows():
            # 该样本的后门模型预测label
            label_o = row["original_backdoorModel_preLabel"]
            # 变量变异标签
            label_change_num = 0
            for mutated_model_global_id in mutated_model_global_id_list:
                model_col_name = f"model_{mutated_model_global_id}"
                label_m = row[model_col_name]
                if label_m != label_o:
                    label_change_num += 1
            lcr_sample = round(label_change_num / len(mutated_model_global_id_list),4)
            lcr_samples.append(lcr_sample)
        avg_lcr = np.mean(lcr_samples)
        data_dict[class_id] = avg_lcr
    sorted_res_1 = dict(sorted(data_dict.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def precision_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num:int):
    '''从小到大'''
    # {class_id(int):[precison_1,precision_2,..precision_500]}
    data_dict = defaultdict(list)
    # ground truth label
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(class_num):
            measure = report[str(class_i)]["precision"]
            data_dict[class_i].append(measure)
    # {class_id(int):avg_precision}
    dict_2 = {}
    for class_id in range(class_num):
        dict_2[class_id] = round(np.mean(data_dict[class_id]),4)
    sorted_res_1 = dict(sorted(dict_2.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def entropy_metrics(df:pd.DataFrame,mutated_model_global_id_list:list,class_num:int):
    '''从大到小'''
    # {class_id(int):avg_LCR}
    data_dict = defaultdict(list)
    for class_id in range(class_num):
        # 抽取出某个类别的df
        class_df = df.loc[df["GT_label"]==class_id]
        # 遍历该类别所有的样本
        entropy_samples = []
        for row_idx,row in class_df.iterrows():
            label_m_list = []
            for mutated_model_global_id in mutated_model_global_id_list:
                model_col_name = f"model_{mutated_model_global_id}"
                label_m = row[model_col_name]
                label_m_list.append(label_m)
            entropy_sample = entropy(label_m_list)
            entropy_samples.append(entropy_sample)
        avg_entropy = round(np.mean(entropy_samples),4)
        data_dict[class_id] = avg_entropy
    sorted_res_1 = dict(sorted(data_dict.items(), key=lambda x: x[1],reverse=True)) # 降序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def look_res():
    FP_res = joblib.load(os.path.join(exp_root_dir,"Exp_Results","ClassRank",dataset_name,model_name,attack_name,str(mutation_rate),"FP.joblib"))
    print(FP_res)

if __name__ == "__main__":
    
    # one-scence
    '''
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    mutation_rate = 0.001
    target_class = 3
    start_time = time.perf_counter()
    one_scene(dataset_name,model_name,attack_name,mutation_rate=mutation_rate, metric="FP")
    # look_res()
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
    '''

    # all-scence
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_name = "discussion_mutation_rate"
    exp_time = get_formattedDateTime()
    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    target_class = 3
    mutation_rate_list = [0.001,0.005,0.007,0.01,0.03,0.05,0.07,0.09,0.1]

    print("exp_root_dir:",exp_root_dir)
    print("exp_name:","discussion_mutation_rate")
    print("exp_time:",exp_time)
    print("dataset_name_list:",dataset_name_list)
    print("model_name_list:",model_name_list)
    print("attack_name_list:",attack_name_list)
    print("mutation_rate_list:",mutation_rate_list)
    print("target class:",target_class)

    all_start_time = time.perf_counter()
    # 三层嵌套字典
    tree = lambda: defaultdict(tree)
    tree_data = tree()
    
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                for mutation_rate in mutation_rate_list:
                    print(f"\n{dataset_name}|{model_name}|{attack_name}|mutation_rate:{mutation_rate}")
                    # save_dir = os.path.join(exp_root_dir,"Exp_Results","ClassRankNew",dataset_name,model_name,attack_name,str(mutation_rate))
                    # os.makedirs(save_dir,exist_ok=True)
                    result = one_scene(dataset_name,model_name,attack_name,
                                       mutation_rate=mutation_rate, metric="FP",save_dir=None)
                    # 现在你可以直接添加键值对，如果不存在会自动创建新的层级
                    tree_data[dataset_name][model_name][attack_name][mutation_rate] = result
    tree_data_save_dir = os.path.join(exp_root_dir,"Exp_Results",exp_name)
    os.makedirs(tree_data_save_dir,exist_ok=True)
    tree_data_save_file_name = "results.json"
    record_save_path = os.path.join(tree_data_save_dir,tree_data_save_file_name)
    with open(record_save_path, 'w') as f:
        json.dump(tree_data, f)
    print(f"results json is saved in {record_save_path}")
    all_end_time = time.perf_counter()
    all_cost_time = all_end_time - all_start_time
    hours, minutes, seconds = convert_to_hms(all_cost_time)
    print(f"all-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")


