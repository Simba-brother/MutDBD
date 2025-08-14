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

if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "GTSRB"
    model_name = "DenseNet"
    attack_name = "IAD"
    target_class = 3
    class_rank = get_classes_rank_v2(exp_root_dir,dataset_name,model_name,attack_name)
    target_class_rank_ratio = round((class_rank.index(target_class)+1) / len(class_rank),3)
    print("END")