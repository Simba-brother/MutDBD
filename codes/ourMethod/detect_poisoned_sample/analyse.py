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
from codes.utils import entropy
from codes.common.eval_model import EvalModel

def cal_entropy(df):
    sample_count = 0
    avg_entropy = 0
    for row in df.iterrows():
        pred_label_list = list(row[1][:-1])
        avg_entropy += entropy(pred_label_list)
        sample_count += 1
    avg_entropy = round(avg_entropy/sample_count,4)
    return avg_entropy

def cal_LCR(df):
    
    
    sample_count = 0
    avg_LCR = 0
    for row in df.iterrows():
        change_num = 0
        o_label = row[1]["origin_model"]
        m_label_list = list(row[1][:-1])
        for m_label in m_label_list:
            if m_label != o_label:
                change_num+=1
        lcr =  change_num/len(m_label_list)
        avg_LCR += lcr
        sample_count += 1
    avg_LCR = round(avg_LCR/sample_count,4)
    return avg_LCR



def data_struct_convertor(pred_label_dict):
    data_struct_convert = {}
    model_count = 0
    for operator in config.mutation_name_list:
        for model_i in range(config.mutation_model_num):
            data_struct_convert[f"model_{model_count}"] = pred_label_dict[operator][model_i]
            model_count += 1
    return data_struct_convert

def look_entropy(pred_label_dict,isPoisoned_groundTruth):
    data_struct_converted = data_struct_convertor(pred_label_dict)
    df = pd.DataFrame(data_struct_converted)
    df["isPoisoned_groundTruth"] = isPoisoned_groundTruth
    df_clean = df.loc[df['isPoisoned_groundTruth'] == True]
    df_poisoned = df.loc[df['isPoisoned_groundTruth'] == False]

    clean_entropy = cal_entropy(df_clean)
    poisoned_entropy = cal_entropy(df_poisoned)
    
    return clean_entropy,poisoned_entropy


def look_LCR(pred_label_dict,isPoisoned_groundTruth:list):
    "label change ratio"
    data_struct_converted = data_struct_convertor(pred_label_dict)
    df = pd.DataFrame(data_struct_converted)
    df["isPoisoned_groundTruth"] = isPoisoned_groundTruth

    device = torch.device(f"cuda:{config.gpu_id}")
    e = EvalModel(backdoor_model,suspiciousClassesDataset,device)
    origin_model_pred_label_list = e.get_pred_labels()
    df["origin_model"] = origin_model_pred_label_list

    df_clean = df.loc[df['isPoisoned_groundTruth'] == True]
    df_poisoned = df.loc[df['isPoisoned_groundTruth'] == False]

    clean_LCR = cal_LCR(df_clean)
    poisoned_LCR = cal_LCR(df_poisoned)
    
    return clean_LCR,poisoned_LCR



def main(analyse_metric_name:str):
    '''
    analyse_metric_name:str
        Entropy | LCR
    '''
    data = {}
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
        if analyse_metric_name == "Entropy":
            clean_metric,poisoned_metric = look_entropy(pred_label_dict,isPoisoned_groundTruth)
        elif analyse_metric_name == "LCR":
            clean_metric,poisoned_metric = look_LCR(pred_label_dict,isPoisoned_groundTruth)
        data[ratio] = {"clean":clean_metric,"poisoned":poisoned_metric}
    return data

if __name__ == "__main__":

    # 进程名称
    proctitle = f"DetectPoisonedSamples_analyse|{config.dataset_name}|{config.model_name}|{config.attack_name}"
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

    analyse_metric_name="Entropy"
    data = main(analyse_metric_name)
    save_dir = os.path.join(
        config.exp_root_dir,
        "DetectPoisonedSamples_analyse",
        config.dataset_name,
        config.model_name,
        config.attack_name,
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = f"{analyse_metric_name}.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(data,save_path)
    print(f"数据保存在:{save_path}")
    