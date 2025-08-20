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
from codes.utils import priorityQueue_2_list,nested_defaultdict,entropy, defaultdict_to_dict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scienceplots

def get_top_k_global_ids(df:pd.DataFrame,top_k=50,trend="bigger",mutation_nums = 500):
    '''
    mutation_nums:所有变异算子变异模型个数
    '''
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    GT_labels = df["GT_label"]
    preLabels_o = df["original_backdoorModel_preLabel"]
    report_o = classification_report(GT_labels,preLabels_o,output_dict=True,zero_division=0)
    for m_i in range(mutation_nums):
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
    print(target_class_rank_ratio)


    # 保存数据
    # save_dir = os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name)
    # os.makedirs(save_dir,exist_ok=True)
    # save_file_name = "res.joblib"
    # save_path = os.path.join(save_dir,save_file_name)
    # joblib.dump(res,save_path)
    # print("保存:",save_path)


def recall_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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

def FP_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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

def f1_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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

def LCR_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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

def precision_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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


def entropy_metrics(df:pd.DataFrame,mutated_model_global_id_list:list):
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
    if metric_name == "FP":
        # 从大到小
        _res = FP_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "precision":
        # 从小到大
        _res = precision_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "Recall":
        # 从小到大
        _res = recall_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "F1":
        # 从小到大
        _res = f1_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "LCR":
        # 从小到大
        _res = LCR_metrics(df_predicted_labels,top_mutated_model_id_list)
    elif metric_name == "Entropy":
        # 从大到小
        _res = entropy_metrics(df_predicted_labels,top_mutated_model_id_list)
    else:
        return None
    return _res

def discussion_rate(mutation_rate):
    '''
    讨论不同变异率对类排序影响
    '''
    
    
    csv_path = os.path.join(exp_root_dir,"EvalMutationToCSV",dataset_name,model_name,attack_name,str(mutation_rate),"preLabel.csv")
    preLabel_df = pd.read_csv(csv_path)

    mutated_model_id_list = get_top_k_global_ids(preLabel_df,top_k=50,trend="bigger")
    # mutated_model_id_list = list(range(50))
    data_stuct_1 = nested_defaultdict(2,int)
    for m_i in mutated_model_id_list:
        pre_labels = preLabel_df[f"model_{m_i}"]
        gt_labels = preLabel_df[f"GT_label"]
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
    return target_class_rank_ratio

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


def compare_WTL(our_list, baseline_list,expect:str, method:str):
    ans = ""
    # 计算W/T/L
    # Wilcoxon:https://blog.csdn.net/TUTO_TUTO/article/details/138289291
    # Wilcoxon：主要来判断两组数据是否有显著性差异。
    if method == "wilcoxon": # 配对
        statistic, p_value = wilcoxon(our_list, baseline_list) # statistic:检验统计量
    elif method == "mannwhitneyu": # 不配对
        statistic, p_value = mannwhitneyu(our_list, baseline_list) # statistic:检验统计量
    elif method == "ks_2samp":
        statistic, p_value = ks_2samp(our_list, baseline_list) # statistic:检验统计量
    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(our_list, baseline_list)
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


def draw_box():
    # 加载数据
    read_data = joblib.load(os.path.join(exp_root_dir,"实验结果","disscution_mutation_rate_for_class_rank.pkl"))
    # {mutation_rate:list}
    conver_data = defaultdict(list)
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
            class_num = get_classNum(dataset_name)
            for model_name in ["ResNet18","VGG19","DenseNet"]:
                for attack_name in ["BadNets","IAD","Refool","WaNet"]:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    conver_data[m_rate].append(read_data[dataset_name][model_name][attack_name][m_rate])
    data = []
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        data.append(conver_data[m_rate])
    # 设置IEEE/Science风格的绘图参数
    plt.style.use(['science','ieee'])
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'stix'


    labels = ['1%', '3%', '5%', '7%', '9%', '10%']

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制箱线图
    boxplot = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    # 自定义箱线图外观
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.7, linewidth=1.5)
        
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linewidth=1.5, linestyle='--')
        
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=1.5)
        
    for median in boxplot['medians']:
        median.set(color='red', linewidth=2)
        
    for mean in boxplot['means']:
        mean.set(marker='o', markerfacecolor='green', markeredgecolor='green', markersize=8)

    # # 计算并标注中位值和均值
    # for i, d in enumerate(data):
    #     median = np.median(d)
    #     mean = np.mean(d)
    #     # 标注中位数
    #     ax.text(i+1, median, f'{median:.3f}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    #     # 标注均值
    #     ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='top', fontsize=9, color='green', fontweight='bold')

    # 添加标签和标题
    ax.set_xlabel('Mutation Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Rank Ratio Distribution at Different Mutation Rates', fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    # plt.show()
    plt.savefig("imgs/1.png")

if __name__ == "__main__":

    # exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    # target_class = 3
    # scence_id = 0
    # res = nested_defaultdict(4,float)
    # for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
    #     class_num = get_classNum(dataset_name)
    #     for model_name in ["ResNet18","VGG19","DenseNet"]:
    #         for attack_name in ["BadNets","IAD","Refool","WaNet"]:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
    #                 print(f"{dataset_name}|{model_name}|{attack_name}|{m_rate}")
    #                 rank_rate = discussion_rate(m_rate)
    #                 res[dataset_name][model_name][attack_name][m_rate] = rank_rate
    # _res = defaultdict_to_dict(res)
    # save_dir = os.path.join(exp_root_dir,"实验结果")
    # save_file_name = "disscution_mutation_rate_for_class_rank.pkl"
    # save_path = os.path.join(save_dir,save_file_name)
    # joblib.dump(_res,save_path)
    # print("结果保存在:",save_path)



    # exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    # target_class = 3
    # mutation_rate = 0.01
    # print("target_class:",target_class)
    # print("mutation_rate:",mutation_rate)
    # FP_list = []
    # precision_list = []
    # Recall_list = []
    # F1_list = []
    # Entropy_list = []
    # for metric_name in ["FP","precision","Recall","F1","Entropy"]:
    #     print(metric_name)
    #     for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
    #         class_num = get_classNum(dataset_name)
    #         for model_name in ["ResNet18","VGG19","DenseNet"]:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             for attack_name in ["BadNets","IAD","Refool","WaNet"]:
    #                 print(f"{dataset_name}|{model_name}|{attack_name}")
    #                 _res = discussion_metric(metric_name)
    #                 if metric_name == "FP":
    #                     FP_list.append(_res["target_class_rank_ratio"])
    #                 if metric_name == "precision":
    #                     precision_list.append(_res["target_class_rank_ratio"])
    #                 if metric_name == "Recall":
    #                     Recall_list.append(_res["target_class_rank_ratio"])
    #                 if metric_name == "F1":
    #                     F1_list.append(_res["target_class_rank_ratio"])
    #                 if metric_name == "Entropy":
    #                     Entropy_list.append(_res["target_class_rank_ratio"])

    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    draw_box()

    # fp_count = 0
    # precision_count = 0
    # recall_count = 0
    # f1_count = 0
    # entropy_count = 0                       
    # for scence_idx in range(len(FP_list)):
    #     fp = FP_list[scence_idx]
    #     precision =  precision_list[scence_idx]
    #     recall = Recall_list[scence_idx]
    #     f1 = F1_list[scence_idx]
    #     e = Entropy_list[scence_idx]
    #     min_value = min(fp,precision,recall,f1,e)
    #     if fp == min_value:
    #         fp_count += 1
    #     if precision == min_value:
    #         precision_count += 1
    #     if recall == min_value:
    #         recall_count += 1
    #     if f1 == min_value:
    #         f1_count += 1
    #     if e == min_value:
    #         entropy_count += 1
    


    # avg_FP = round(np.mean(FP_list),3)
    # avg_precision = round(np.mean(precision_list),3)
    # avg_Recall = round(np.mean(Recall_list),3)
    # avg_F1 = round(np.mean(F1_list),3)
    # avg_Entropy = round(np.mean(Entropy_list),3)

    # wtl_precison = compare_WTL(FP_list,precision_list,"small","wilcoxon")
    # wtl_recall = compare_WTL(FP_list,Recall_list,"small","wilcoxon")
    # wtl_f1 = compare_WTL(FP_list,F1_list,"small","wilcoxon")
    # wtl_entropy = compare_WTL(FP_list,Entropy_list,"small","wilcoxon")

    # print("FP_list:",FP_list)
    # print("precision_list:",precision_list)
    # print("Recall_list:",Recall_list)
    # print("F1_list:",F1_list)
    # print("Entropy_list:",Entropy_list)

    # print("fp_count:",fp_count)
    # print("precision_count:",fp_count)
    # print("recall_count:",recall_count)
    # print("f1_count:",f1_count)
    # print("entropy_count:",entropy_count)

    # print("avg_FP:",avg_FP)
    # print("avg_precision:",avg_precision)
    # print("avg_Recall:",avg_Recall)
    # print("avg_F1:",avg_F1)
    # print("avg_Entropy:",avg_Entropy)

    # print("wtl_precison:",wtl_precison)
    # print("wtl_recall:",wtl_recall)
    # print("wtl_f1:",wtl_f1)
    # print("wtl_entropy:",wtl_entropy)

    # with_ClassRank_list = [56.2,22.9,24.1,22.2,8.8,17,387.9,323.6,194.4,22.9,415.1,34.3,18.2,19.6,12.2,15.4,4.5,2.8,117.1,26.6,33.1,23.5,38.4,7.5,1.8,3.8,376.3,24.5,3.6]
    # no_ClassRank_list = [117.3,80.3,265,78,36.7,36.7,1248.4,793,263.1,59.3,769.7,153.7,28.6,24.2,15.3,19.2,11.6,20.1,235.6,33.9,39.2,28.3,46.8,24.2,46.1,46,846.1,132.5,40.5]
    # assert len(with_ClassRank_list) == len(no_ClassRank_list), "没有配对成功"
    # wtl_ans = caculate_WTL(with_ClassRank_list,no_ClassRank_list,"small")
    # avg_with_classRank_list = round(sum(with_ClassRank_list)/len(with_ClassRank_list),3)
    # avg_no_classRank_list = round(sum(no_ClassRank_list)/len(no_ClassRank_list),3)
    # print(wtl_ans)
    # print(avg_with_classRank_list)
    # print(avg_no_classRank_list)

    # exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    # dataset_name_list = ["CIFAR10","GTSRB","ImageNet2012_subset"]
    # model_name_list = ["ResNet18","VGG19","DenseNet"]
    # attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    # mutation_rate = 0.01
    # target_class = 3
    # for dataset_name in dataset_name_list:
    #     class_num = get_classNum(dataset_name)
    #     for model_name in model_name_list:
    #         for attack_name in attack_name_list:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             print(f"{dataset_name}|{model_name}|{attack_name}")
    #             main()
    #             # look()
    # print("END")

