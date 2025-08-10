import os
import joblib
import torch

'''
从硬盘加载部分实验节点的结果
'''
def get_classes_rank_v2(exp_root_dir,dataset_name,model_name,attack_name):
    data_path = os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,"res.joblib")
    data = joblib.load(data_path)
    return data["class_rank"]

def get_backdoor_info(exp_root_dir,dataset_name,model_name,attack_name):
    # 后门信息
    backdoor_data = torch.load(os.path.join(exp_root_dir, "ATTACK",
                            dataset_name, model_name, attack_name,
                            "backdoor_data.pth"), map_location="cpu")
    return backdoor_data