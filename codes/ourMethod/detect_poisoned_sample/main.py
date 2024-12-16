'''
检测中毒样本
'''
import setproctitle
from codes import config
import torch
import os
from codes.scripts.dataset_constructor import *
import joblib
import pandas as pd
from codes.utils import entropy,priorityQueue_2_list
import queue




def detect(df):
    q = queue.PriorityQueue() # 熵越小越可能为poisoned,队头
    row_id = 0
    for row in df.iterrows():
        e = entropy(list(row[:-1]))
        q.put((e, row[-1],row_id))
        row_id += 1
    priority_list = priorityQueue_2_list(q)

    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    f1_list = []

    gt_TP = sum(list(df["gt_isPoisoned"]))
    for cut_off in cut_off_list:
        end = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:end]
        TP = 0
        FP = 0
        for item in prefix_priority_list:
            gt = item[1]
            if gt == True:
                TP += 1
            else:
                FP += 1
        precision = round(TP/(TP+FP),4)
        recall = round(TP/gt_TP,4)
        f1 = round(2 * precision*recall / (precision*recall),4)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list,recall_list,f1_list

def main():
    data_struct_convert = {}
    for ratio in config.fine_mutation_rate_list:
        data_struct_convert[ratio] = {}
        model_count = 0
        for operator in config.mutation_name_list:
            for model_i in range(config.mutation_model_num):
                data_struct_convert[ratio][f"model_{model_count}"] = pred_label_ans[ratio][operator][model_i]
                model_count += 1
    res = {}
    for ratio in config.fine_mutation_rate_list:
        # 每一列为一个变异模型在候选集上的预测label
        df = pd.DataFrame(data_struct_convert[ratio])
        df["gt_isPoisoned"] = gt_isPoisoned
        precision_list,recall_list,f1_list = detect(df)
        res[ratio] = {
            "precision_list":precision_list,
            "recall_list":recall_list,
            "f1_list":f1_list,
        }
    return res

if __name__ == "__main__":

    pred_label_ans = joblib.load(os.path.join(
        config.exp_root_dir,
        "EvalMutationResult",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "pred_label_ans.data"
    ))

    gt_isPoisoned = joblib.load(os.path.join(
        config.exp_root_dir,
        "EvalMutationResult",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "gt_isPoisoned.data"
    ))

    res = main()

    # 保存数据
    




    pass