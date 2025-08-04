# 用于计算class rank
import os
import queue
import joblib

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from scipy.stats import wilcoxon,mannwhitneyu,ks_2samp
from cliffs_delta import cliffs_delta

from codes.utils import priorityQueue_2_list,nested_defaultdict,defaultdict_to_dict


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
    
    dataset_name = "ImageNet2012_subset"
    class_num = get_classNum(dataset_name)
    model_name = "DenseNet"
    attack_name = "BadNets"
    target_class = 3
    for mutation_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        print("变异率:",mutation_rate)
        main()

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

