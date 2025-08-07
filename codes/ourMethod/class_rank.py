# 用于计算class rank
import os
import queue
import joblib
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from scipy.stats import wilcoxon,mannwhitneyu,ks_2samp
from cliffs_delta import cliffs_delta
from codes.utils import priorityQueue_2_list,nested_defaultdict,entropy


def get_top_k_global_ids(df:pd.DataFrame,top_k=50,trend="bigger"):
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    GT_labels = df["GT_label"]
    preLabels_o = df["original_backdoorModel_preLabel"]
    report_o = classification_report(GT_labels,preLabels_o,output_dict=True,zero_division=0)
    for m_i in range(500):
        col_name = f"model_{m_i}"
        preLabel_m = df[col_name]
        report_m = classification_report(GT_labels,preLabel_m,output_dict=True,zero_division=0)
        acc_dif = abs(report_o["accuracy"] - report_m["accuracy"])
        if trend == "smaller":
            item = (acc_dif, m_i)
        else:
            item = (-acc_dif, m_i)
        q.put(item)
    
    priority_list = priorityQueue_2_list(q)
    selected_m_i_list = [m_i for priority, m_i in  priority_list[0:top_k]]
    return selected_m_i_list


def main():
    '''获得class rank list and rank top'''
    # 加载变异模型的预测标签
    df_predicted_labels = pd.read_csv(os.path.join(
            exp_root_dir,
            "EvalMutationToCSV",
            dataset_name,
            model_name,
            attack_name,
            str(mutation_rate),
            "preLabel.csv"))
    # 加载top50变异模型id
    mutated_model_id_list = get_top_k_global_ids(df_predicted_labels,top_k=50,trend="bigger")

    data_stuct_1 = nested_defaultdict(2,int)
    for m_i in mutated_model_id_list:
        pre_labels = df_predicted_labels[f"model_{m_i}"]
        gt_labels = df_predicted_labels[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(class_num):
            data_stuct_1[m_i][class_i] = np.sum(cm[:,class_i]) - cm[class_i][class_i]

    data_stuct_2 = nested_defaultdict(1,int)
    for class_i in range(class_num):
        for m_i in mutated_model_id_list:
            data_stuct_2[class_i] += data_stuct_1[m_i][class_i]
    sorted_res_1 = dict(sorted(data_stuct_2.items(), key=lambda x: x[1],reverse=True))
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    print(dataset_name,model_name,attack_name)
    print(res)
    print("="*10)

    # 保存数据
    # save_dir = os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name)
    # os.makedirs(save_dir,exist_ok=True)
    # save_file_name = "res.joblib"
    # save_path = os.path.join(save_dir,save_file_name)
    # joblib.dump(res,save_path)
    # print("保存:",save_path)


def recall_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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
    # 按照avg_LCR对class_id进行排序（降序），recall越大的class越可疑
    sorted_res_1 = dict(sorted(dict_2.items(), key=lambda x: x[1],reverse=True)) # 降序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def f1_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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
    # 按照avg_f1对class_id进行排序（降序），f1越小的class越可疑
    sorted_res_1 = dict(sorted(dict_2.items(), key=lambda x: x[1],reverse=False)) # 降序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def LCR_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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
    # 按照avg_lcr对class_id进行排序（升序），lcr越小的class越可疑
    sorted_res_1 = dict(sorted(data_dict.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def entropy_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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
    # 按照avg_entropy对class_id进行排序（升序），entropy越小（越纯）的class越可疑
    sorted_res_1 = dict(sorted(data_dict.items(), key=lambda x: x[1],reverse=False)) # 升序
    class_rank = list(sorted_res_1.keys())
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    res = {
        "class_rank":class_rank,
        "target_class_rank_ratio":target_class_rank_ratio
    }
    return res

def discussion_metric(metric_name:str):
    # 加载变异模型的预测标签
    df_predicted_labels = pd.read_csv(os.path.join(
            exp_root_dir,
            "EvalMutationToCSV",
            dataset_name,
            model_name,
            attack_name,
            str(mutation_rate),
            "preLabel.csv"))
    # 加载top50变异模型id
    top_mutated_model_id_list = get_top_k_global_ids(df_predicted_labels,top_k=50,trend="bigger")
    if metric_name == "Recall":
        _res = recall_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "F1":
        _res = f1_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "LCR":
        _res = LCR_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "Entropy":
        _res = entropy_metrics(df_predicted_labels,top_mutated_model_id_list)
    else:
        return None
    return _res

def look():
    # 保存数据
    data_path = os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,"res.joblib")
    data = joblib.load(data_path)
    print(f"{dataset_name}|{model_name}|{attack_name}")
    print(data)
    return data


def get_classNum(dataset_name):
    class_num = None
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    return class_num


# 计算W/T/L
def caculate_WTL(data_list, baseline_list, expect:str):
    statistic, p_value = wilcoxon(data_list, baseline_list) # statistic:检验统计量
    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(data_list, baseline_list)
    if p_value >= 0.05:
        # 值分布没差别
        ans = "Tie"
        return ans
    else:
        # 值分布有差别
        if expect == "small":
            # 指标越小越好，d越接近-1越好
            if d < 0 and res != "negligible":
                ans = "Win"
            elif d > 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
        else:
            # 指标越大越好，d越接近1越好
            if d > 0 and res != "negligible":
                ans = "Win"
            elif d < 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
    return ans

if __name__ == "__main__":

    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    target_class = 3
    mutation_rate = 0.01
    metric_name = "Entropy"
    print("target_class:",target_class)
    print("mutation_rate:",mutation_rate)
    print("metric_name:",metric_name)
    for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
        class_num = get_classNum(dataset_name)
        for model_name in ["ResNet18","VGG19","DenseNet"]:
            if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                continue
            for attack_name in ["BadNets","IAD","Refool","WaNet"]:
                print(f"{dataset_name}|{model_name}|{attack_name}")
                _res = discussion_metric(metric_name)
                print("rank_rate:",_res["target_class_rank_ratio"])

    # with_ClassRank_list = [56.2,22.9,24.1,22.2,8.8,17,387.9,323.6,194.4,22.9,415.1,34.3,18.2,19.6,12.2,15.4,4.5,2.8,117.1,26.6,33.1,23.5,38.4,7.5,1.8,3.8,376.3,24.5,3.6]
    # no_ClassRank_list = [117.3,80.3,265,78,36.7,36.7,1248.4,793,263.1,59.3,769.7,153.7,28.6,24.2,15.3,19.2,11.6,20.1,235.6,33.9,39.2,28.3,46.8,24.2,46.1,46,846.1,132.5,40.5]
    # assert len(with_ClassRank_list) == len(no_ClassRank_list), "没有配对成功"
    # wtl_ans = caculate_WTL(with_ClassRank_list,no_ClassRank_list,"small")
    # avg_with_classRank_list = round(sum(with_ClassRank_list)/len(with_ClassRank_list),3)
    # avg_no_classRank_list = round(sum(no_ClassRank_list)/len(no_ClassRank_list),3)
    # print(wtl_ans)
    # print(avg_with_classRank_list)
    # print(avg_no_classRank_list)

    # dataset_name_list = ["ImageNet2012_subset"] # ["CIFAR10","GTSRB","ImageNet2012_subset"]
    # model_name_list =  ["DenseNet"] # ["ResNet18","VGG19","DenseNet"]
    # attack_name_list =  ["BadNets"] # ["BadNets","IAD","Refool","WaNet"]
    # mutation_rate = 0.01
    # target_class = 3
    # for dataset_name in dataset_name_list:
    #     class_num = get_classNum(dataset_name)
    #     for model_name in model_name_list:
    #         for attack_name in attack_name_list:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             # main()
    #             # look()
    # print("END")

