
import os
import torch
import joblib
import json
import numpy as np
from mid_data_loader import get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from utils.calcu_utils import compare_WTL,compare_avg


def strip_sampling(entropy_list, poisoned_ids, cut_off:float = 0.4):
    ranked_ids = np.argsort(entropy_list) # 熵越小的idx排越前
    cut = int(len(ranked_ids)*cut_off)
    suspicious_ids = ranked_ids[:cut]
    all_ids = list(range(len(entropy_list)))
    remain_ids = list(set(all_ids) - set(suspicious_ids))
    p_ids = list(set(remain_ids) & set(poisoned_ids))
    PN = len(p_ids)
    sample_res = {
        "PN":PN,
        "remain_ids":remain_ids
    }
    return sample_res

def load_strip_pn_list(dataset_name,model_name,attack_name,poisoned_ids)->list[int]:
    PN_list = []
    for r_seed in r_seed_list:
        entropys_path = os.path.join(exp_root_dir,"Defense/Strip_hardCut/sampling", 
                                     dataset_name, model_name, attack_name, f"exp_{r_seed}", "entropy.pt")
        entropy_list = torch.load(entropys_path,map_location="cpu")
        sample_res = strip_sampling(entropy_list, poisoned_ids, cut_off = 0.4)
        PN_list.append(sample_res["PN"])
    assert len(PN_list) == len(r_seed_list), "数据没匹配"
    return PN_list

def read_ours_result(dataset_name,model_name,attack_name):
    # 读取我们的list
    if dataset_name in ["CIFAR10","GTSRB"]:
        res = joblib.load(os.path.join(exp_root_dir,"Exp_Results","eval_ours_asd",
                                       dataset_name,model_name,attack_name,"res_818.pkl"))
        our_asr_list = res["our_asr_list"]
        our_acc_list = res["our_acc_list"]
        our_pn_list = res["our_p_num_list"]
    else:
        json_path = os.path.join(exp_root_dir,"Exp_Results","eval_ours_asd","ImageNet","res.json")
        with open(json_path,mode="r") as f:
            res = json.load(f)
            our_asr_list = res[dataset_name][model_name][attack_name]["ours"]["asr_list"]
            our_acc_list = res[dataset_name][model_name][attack_name]["ours"]["acc_list"]
            our_pn_list = res[dataset_name][model_name][attack_name]["ours"]["pnum_list"]

    return our_asr_list, our_acc_list, our_pn_list

def one_scene(dataset_name,model_name,attack_name):
    backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
    poisoned_ids = backdoor_data["poisoned_ids"]
    strip_pn_list = load_strip_pn_list(dataset_name,model_name,attack_name,poisoned_ids)
    our_asr_list, our_acc_list, our_pn_list = read_ours_result(dataset_name,model_name,attack_name) 
    ours_pn_avg,strip_pn_avg = compare_avg(our_pn_list,strip_pn_list)
    pn_wtl = compare_WTL(our_pn_list,strip_pn_list, "small","mannwhitneyu")
    print(f"{dataset_name}|{model_name}|{attack_name}|our_avg:{ours_pn_avg}|strip_avg:{strip_pn_avg}|wtl:{pn_wtl}")






if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))

    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                one_scene(dataset_name,model_name,attack_name)