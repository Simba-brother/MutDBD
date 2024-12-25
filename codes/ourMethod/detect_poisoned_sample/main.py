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
from sklearn.metrics import precision_score,recall_score,f1_score


    

            


def detect_by_entropy(df):
    '''
    熵的检测规则是：熵越大，说明该样本在变异模型集上的预测标签越乱，说明样本受到决策边界影响越大。
    我们认为，木马样本受打决策边界的影响较大，所以木马样本应该有较高的熵。
    '''
    q = queue.PriorityQueue() # api是小顶堆
    row_id = 0
    for row in df.iterrows():
        pred_label_list = list(row[1][:-1])
        isPoisoned_gt = row[1][-1]
        e = entropy(pred_label_list)
        q.put((-e, isPoisoned_gt,row_id)) # 熵越大，木马优先级越高
        row_id += 1

    priority_list = priorityQueue_2_list(q)

    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    f1_list = []
    for cut_off in cut_off_list:
        c = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:c]
        remain_priority_list = priority_list[c:]
        pred_list = []
        for _ in range(len(prefix_priority_list)):
            pred_list.append(True)
        for _ in range(len(remain_priority_list)):
            pred_list.append(False)
        gt_list = []
        for item in priority_list:
            gt_list.append(item[1])
        
        # 计算精度
        precision = precision_score(gt_list, pred_list)
        # 计算召回率
        recall = recall_score(gt_list, pred_list)
        # 计算F1分数
        f1 = f1_score(gt_list, pred_list)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list,recall_list,f1_list

def detect(pred_label_dict,isPoisoned_groundTruth):
    data_struct_convert = {}
    model_count = 0
    for operator in config.mutation_name_list:
        for model_i in range(config.mutation_model_num):
            data_struct_convert[f"model_{model_count}"] = pred_label_dict[operator][model_i]
            model_count += 1
    
        
    df = pd.DataFrame(data_struct_convert)
    df["isPoisoned_groundTruth"] = isPoisoned_groundTruth
    precision_list,recall_list,f1_list = detect_by_entropy(df)
    res = {
        "precision_list":precision_list,
        "recall_list":recall_list,
        "f1_list":f1_list,
    }
    return res

if __name__ == "__main__":

    # 进程名称
    proctitle = f"DetectPoisonedSamples|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    print(proctitle)

    # 加载后门模型数据
    backdoor_data_path = os.path.join(config.exp_root_dir, "ATTACK", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_ids =backdoor_data["poisoned_ids"]

    # 加载后门模型中的可疑classes
    suspicious_classes_dict = joblib.load(os.path.join(
        config.exp_root_dir,
        "SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "SuspiciousClasses_SK_Precision.data"
    ))

    for ratio in config.fine_mutation_rate_list:
        # 获得该变异率下suspicious_classes
        suspicious_classes = suspicious_classes_dict[ratio]
        suspiciousClassesDataset = ExtractSuspiciousClassesDataset(poisoned_trainset,suspicious_classes,poisoned_ids)
        isPoisoned_groundTruth = []
        for i in range(len(suspiciousClassesDataset)):
            sample,label,isPoisoned = suspiciousClassesDataset[i]
            isPoisoned_groundTruth.append(isPoisoned)
        
        pred_label_dict = joblib.load(os.path.join(
            config.exp_root_dir,
            "SuspiciousClassesPredLabel",
            config.dataset_name, 
            config.model_name, 
            config.attack_name,
            str(ratio),
            "res.data"
        ))

        res = detect(pred_label_dict,isPoisoned_groundTruth)

        save_dir = os.path.join(
            config.exp_root_dir,
            "DetectPoisonedSamples",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(ratio)
        )
        os.makedirs(save_dir,exist_ok=True)
        save_file_name = "P_R_F1_dict.data"
        save_path = os.path.join(save_dir,save_file_name)
        joblib.dump(res,save_path)
        print(f"数据保存在:{save_path}")
    