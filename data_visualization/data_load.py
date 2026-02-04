
import joblib
import os
from collections import defaultdict
from utils.common_utils import read_yaml
from utils.dataset_utils import get_class_num
from models.model_loader import get_model
import torch
from mid_data_loader import get_backdoor_data,get_class_rank
from datasets.posisoned_dataset import get_all_dataset
from utils.model_eval_utils import eval_asr_acc
from defense.our.sample_select import chose_retrain_set

# 加载数据
config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]
def load_m_rate_scence_class_rank():
    read_data = joblib.load(os.path.join(exp_root_dir,"Exp_Results","disscution_mutation_rate_for_class_rank.pkl"))
    # {mutation_rate:list}
    ans = defaultdict(list)
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
            class_num = get_class_num(dataset_name)
            for model_name in ["ResNet18","VGG19","DenseNet"]:
                for attack_name in ["BadNets","IAD","Refool","WaNet"]:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    ans[m_rate].append(read_data[dataset_name][model_name][attack_name][m_rate])
    return ans

def load_cutoff_data(dataset_name,model_name,attack_name,random_seed, device):
    # 加载两个空的模型结构
    select_model = get_model(dataset_name,model_name)
    defense_model = get_model(dataset_name,model_name)
    # 加载选择模型权重
    selected_state_dict_path = os.path.join(exp_root_dir,"cut_off",dataset_name,model_name,attack_name,f"exp_{random_seed}","best_BD_model.pth")
    select_model.load_state_dict(torch.load(selected_state_dict_path,"cpu"))

    class_rank = get_class_rank(dataset_name,model_name,attack_name)
    # 获得数据
    backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset = get_all_dataset(dataset_name,model_name,attack_name,poisoned_ids)

    

    CutOff_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    PN_rate_list = []
    ASR_list = []
    ACC_list = []
    for cut_off in CutOff_list:
        # 加载防御模型权重
        defense_state_dict_path = os.path.join(exp_root_dir,"cut_off",dataset_name,model_name,attack_name,f"exp_{random_seed}",str(cut_off),"best_defense_model.pth")
        defense_model.load_state_dict(torch.load(defense_state_dict_path,"cpu"))
        choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = chose_retrain_set(select_model, device, 
                      cut_off, poisoned_trainset, poisoned_ids, class_rank=class_rank)
        ASR,ACC = eval_asr_acc(defense_model,filtered_poisoned_testset,clean_test_dataset,device)
        PN_rate_list.append(round(PN/len(poisoned_ids),4))
        ASR_list.append(ASR)
        ACC_list.append(ACC)
        print(f"cut_off完成:{cut_off}")
    return CutOff_list,PN_rate_list,ASR_list,ACC_list


