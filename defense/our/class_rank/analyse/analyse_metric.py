'''分析不同度量对class rank的影响'''
import os
import joblib
from commonUtils import read_yaml
config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]


def one_scene(dataset_name,model_name,attack_name,mutation_rate=0.01): 
    FP_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"FP.joblib"))
    Precision_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"Precision.joblib"))
    Recall_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"Recall.joblib"))
    F1_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"F1.joblib"))
    LCR_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"LCR.joblib"))
    Entropy_res = joblib.load(os.path.join(exp_root_dir,"实验结果","类排序",dataset_name,model_name,attack_name,str(mutation_rate),"Entropy.joblib"))
    print(f'FP:rank_rate:{FP_res["target_class_rank_ratio"]}')
    print(f'Precision:rank_rate:{Precision_res["target_class_rank_ratio"]}')
    print(f'Recall:rank_rate:{Recall_res["target_class_rank_ratio"]}')
    print(f'F1:rank_rate:{F1_res["target_class_rank_ratio"]}')
    print(f'LCR:rank_rate:{LCR_res["target_class_rank_ratio"]}')
    print(f'Entropy:rank_rate:{Entropy_res["target_class_rank_ratio"]}')

if __name__ == "__main__":
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    one_scene(dataset_name,model_name,attack_name)

