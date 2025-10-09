'''
重要
怀疑类别集检测器
'''
import os
import queue
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from sklearn.metrics import classification_report
import joblib

from codes import config
# 得到格式化时间串
from codes.common.time_handler import get_formattedDateTime
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids
from codes.utils import entropy,priorityQueue_2_list,calcu_LCR, nested_defaultdict, defaultdict_to_dict
from codes.common.logging_handler import get_Logger
import matplotlib.pyplot as plt


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
    # 获得类别排名,值越小优先级越大（小根堆）
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    classes_priority = [priority for priority,class_i in priority_list]
    target_class_ranking_percent = round((classes_rank.index(config.target_class_idx)+1)/len(classes_rank),3)

    return classes_rank, target_class_ranking_percent, class_precisionList_dict

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
    if method == "Precision_mean":
        class_rank,rank_rate,class_precisionList_dict = detect_by_precision(df_Label,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Precision_var":
        class_rank,rank_rate,class_precisionList_dict = detect_by_precision(df_Label,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Loss_mean":
        class_rank,rank_rate = detect_by_loss(df_CELoss,class_num,mutated_model_global_id_list,stat_name="mean")
    if method == "Loss_var":
        class_rank,rank_rate = detect_by_loss(df_CELoss,class_num,mutated_model_global_id_list,stat_name="var")
    if method == "Recall_mean":
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
    return class_rank,rank_rate,class_precisionList_dict



def f(dataset_name,model_name:str,attack_name:str,class_num:int,mutated_rate,detect_method):
    save_dir = os.path.join(
        config.exp_root_dir,
        "ClassRank",
        dataset_name,
        model_name,
        attack_name,
        str(mutated_rate),
        detect_method
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "ClassRank.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    df_Label = load_df(dataset_name,model_name,attack_name,mutated_rate,"preLabel")
    # 选择出top50变异模型
    mutated_model_global_id_list = get_top_k_global_ids(df_Label,top_k=50,trend="bigger")
    data = nested_defaultdict(5)
    data[dataset_name][model_name][attack_name][mutated_rate]["top50_model_id_list"] = mutated_model_global_id_list
    data = defaultdict_to_dict(data)
    save_dir = os.path.join(config.exp_root_dir, "实验结果")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "ImageNet_grid.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    
    joblib.dump(data,save_path)
    print(save_path)
    print("END")

    # class_rank,target_class_ranking_percent, class_precisionList_dict = detect_method_pool(df_Label,None,class_num,mutated_model_global_id_list,detect_method)
    # print(f"class_rank:{class_rank}")
    # print(f"target_class_ranking_percent:{str(target_class_ranking_percent)}")
    # data = {
    #     "class_rank":class_rank,
    #     "target_class_ranking_percent":target_class_ranking_percent
    # }
    # joblib.dump(data,save_path)
    # print("save_path",save_path)

def main():
    '''
    CIFAR10|GTSRB
    '''
    data = {}
    dataset_name_list = ["CIFAR10","GTSRB"] # config.cur_dataset_name_list
    model_name_list =  ["ResNet18","VGG19"] # config.cur_model_name_list
    attack_name_list = ["BadNets"] # config.cur_attack_name_list
    mutation_rate_list = [0.01] # config.fine_mutation_rate_list
    detect_method_list = ["Precision_mean"]
    # detect_method_list = ["Precision_mean","Precision_var","Loss_mean","Loss_var","Recall_mean","Recall_var",
    #                     "Entropy_model_mean","Entropy_model_var","Entropy_sample_mean","Entropy_sample_var",
    #                     "LCR_model_mean","LCR_model_var","LCR_sample_mean","LCR_sample_var"
    #                     ]
    for dataset_name in dataset_name_list:
        print(f"dataset_name:{dataset_name}")
        data[dataset_name] = {}
        if dataset_name == "CIFAR10":
            class_num = 10
        if dataset_name == "GTSRB":
            class_num = 43
        for model_name in model_name_list:
            print(f"\tmodel_name:{model_name}")
            data[dataset_name][model_name] = {}
            for attack_name in attack_name_list:
                print(f"\t\tattack_name:{attack_name}")
                data[dataset_name][model_name][attack_name] = {}
                for mutated_rate in mutation_rate_list:
                    print(f"\t\t\tmutated_rate:{str(mutated_rate)}")
                    data[dataset_name][model_name][attack_name][mutated_rate] = {}
                    # 预测标签df
                    df_Label = load_df(dataset_name,model_name,attack_name,mutated_rate,"preLabel")
                    # 选择出top50变异模型
                    mutated_model_global_id_list = get_top_k_global_ids(df_Label,top_k=50,trend="bigger")
                    data[dataset_name][model_name][attack_name][mutated_rate]["top50_model_id_list"] = mutated_model_global_id_list
                    # if dataset_name in ["CIFAR10","GTSRB"]:
                    #     df_CELoss = load_df(dataset_name,model_name,attack_name,mutated_rate,"CELoss")
                    # else:
                    #     df_CELoss = None
                    # for detect_method in detect_method_list:
                    #     print(f"\t\t\t\tdetect_method:{detect_method}")
                    #     class_rank,target_class_ranking_percent,class_dataList_dict = detect_method_pool(df_Label,df_CELoss,class_num,mutated_model_global_id_list,detect_method)
                    #     print(f"\t\t\t\t\tclass_rank:{class_rank}")
                    #     print(f"\t\t\t\t\ttarget_class_ranking_percent:{str(target_class_ranking_percent)}")
                    #     data[dataset_name][model_name][attack_name][mutated_rate][detect_method] = {}
                    #     data[dataset_name][model_name][attack_name][mutated_rate][detect_method]["class_rank"] = class_rank
                    #     data[dataset_name][model_name][attack_name][mutated_rate][detect_method]["target_class_ranking_percent"] = target_class_ranking_percent
                    #     data[dataset_name][model_name][attack_name][mutated_rate][detect_method]["class_dataList_dict"] = class_dataList_dict
                    
    save_dir = os.path.join(config.exp_root_dir, "实验结果")
    save_file_name = "grid.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(data,save_path)
    print(save_path)
    print("END")


def look_res():
    '''
    根据需要看结果
    '''
    grid = joblib.load(os.path.join(config.exp_root_dir,"grid.joblib"))
    mutated_rate = 0.01
    for measure_name in ["Precision_mean","Precision_var","Loss_mean","Loss_var","Recall_mean","Recall_var",
                        "Entropy_model_mean","Entropy_model_var","Entropy_sample_mean","Entropy_sample_var",
                        "LCR_model_mean","LCR_model_var","LCR_sample_mean","LCR_sample_var"
                        ]:
        print(measure_name)
        for dataset_name in config.cur_dataset_name_list: # 2个
            for model_name in config.cur_model_name_list: # 3个
                for attack_name in config.cur_attack_name_list: # 4个
                    precent = grid[dataset_name][model_name][attack_name][mutated_rate][measure_name]["target_class_ranking_percent"]
                    print(f"{round(precent*100,1)}%",end='\t')
                print("\n")
        print("="*30)

def look_res_2():
    '''
    根据需要看结果
    '''
    data = joblib.load(os.path.join(config.exp_root_dir,"grid.joblib"))
    
    threshold = 0.75
    for measure_name in ["Precision_mean","Loss_mean","Recall_mean","Precision_var","Loss_var","Recall_var",
                        "Entropy_model_mean","Entropy_model_var","Entropy_sample_mean","Entropy_sample_var",
                        "LCR_model_mean","LCR_model_var","LCR_sample_mean","LCR_sample_var"
                        ]:
        total = 24
        count  = 0
        for mutated_rate in config.fine_mutation_rate_list:
            for dataset_name in config.cur_dataset_name_list: # 2个
                for model_name in config.cur_model_name_list: # 3个
                    for attack_name in config.cur_attack_name_list: # 4个
                        percent = data[dataset_name][model_name][attack_name][mutated_rate][measure_name]["target_class_ranking_percent"]
                        if percent >= threshold:
                            count += 1
            avg_count = round(count/len(config.fine_mutation_rate_list),3)
        print(f"{avg_count}/{total}")


def data_visualization():
    # 加载数据
    data = joblib.load(os.path.join(config.exp_root_dir,"实验结果","grid.joblib"))
    class_datalist_dict = data["CIFAR10"]["ResNet18"]["BadNets"][0.01]["Precision_mean"]["class_dataList_dict"]
    box_data = []
    class_i_list = list(range(10))
    for class_i in class_i_list:
         box_data.append(class_datalist_dict[class_i])
    plt.figure(figsize=(10, 6))  # 设置画布大小

    # 绘制箱线图
    box = plt.boxplot(box_data, 
                    patch_artist=True,  # 允许填充颜色
                    showmeans=False,     # 先不显示默认均值标记
                    labels=class_i_list)

    # 隐藏中位线
    for element in ['medians']:
        for line in box[element]:
            line.set_visible(False)  # 将中位数设置为不可见

    # 自定义箱体颜色 - 类别3为红色，其他为绿色
    colors = []
    for i in range(len(box_data)):
        if i == 3:  # 类别3（索引3）设为红色
            colors.append('red')
        else:       # 其他类别设为绿色
            colors.append('green')
    # 应用颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # 设置透明度

    # 设置箱线图线条样式
    # plt.setp(box['medians'], color='black', linewidth=2)  # 中位数线
    plt.setp(box['whiskers'], color='gray', linestyle='--') 
    plt.setp(box['caps'], color='gray', linewidth=2)

    # 计算并添加均值线
    means = [np.mean(d) for d in box_data]  # 计算每个类别的均值
    for i, mean in enumerate(means):
        # 在每个箱线图位置添加均值横线
        plt.hlines(mean, 
                xmin=i+0.8, 
                xmax=i+1.2, 
                colors='black', 
                linewidth=3,
                label='mean' if i == 0 else "")  # 仅添加一次图例

    # 添加标签和标题
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy distribution of different categories', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线

    # 添加图例
    plt.legend(loc="center right")

    # 优化布局并显示
    plt.tight_layout()
    plt.show()
    plt.savefig("imgs/detect_target_class.png")

if __name__ == "__main__":
    # dataset_name = "ImageNet2012_subset"
    # model_name = "DenseNet"
    # attack_name = "BadNets"
    # class_num = 30
    # mutated_rate = 0.01 # [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
    # '''
    # ["Precision_mean","Precision_var","Loss_mean","Loss_var","Recall_mean","Recall_var" , 
    # "Entropy_model_mean","Entropy_model_var","Entropy_sample_mean","Entropy_sample_var",
    # "LCR_model_mean","LCR_model_var","LCR_sample_mean","LCR_sample_var"]
    # '''
    # detect_method = "Precision_mean"
    # if dataset_name in ["CIFAR10","GTSRB"]:
    #     main()  
    # elif dataset_name == "ImageNet2012_subset":
    #     f(dataset_name,model_name,attack_name,class_num,mutated_rate,detect_method)
        
    # look_res()
    # main()
    # data_visualization()


    dataset_name = "ImageNet2012_subset"
    model_name = "DenseNet"
    attack_name = "IAD"
    class_num = 30
    mutated_rate = 0.01
    detect_method = "Precision_mean"
    f(dataset_name,model_name,attack_name,class_num,mutated_rate,detect_method)
    



