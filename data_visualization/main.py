import os
import json
import torch
from data_visualization.data_load import load_m_rate_scence_class_rank,load_cutoff_data
from data_visualization.draw import draw_box,draw_cutoff
from collections import defaultdict

def vis_discussion_1():
    '''不同变异率下所有场景的箱线图'''
    '''论文插图：不同变异率对FP class rank的影响'''
    # data = load_m_rate_scence_class_rank()
    json_path = os.path.join(exp_root_dir,"Exp_Results","discussion_mutation_rate","results.json")
    with open(json_path, mode="r") as f:
        res = json.load(f)
    
    dataset_name_list = ["CIFAR10","GTSRB","ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    rate_list = [0.001,0.005,0.007,0.01,0.03,0.05,0.07,0.09,0.1]
    data = defaultdict(list)
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                continue
            for attack_name in attack_name_list:
                for rate in rate_list:
                    data[rate].append(res[dataset_name][model_name][attack_name][str(rate)]["rank_ratio"])

    save_dir = "imgs/discussion"
    save_file_name = "discussion_1.png"
    save_path = os.path.join(save_dir, save_file_name)
    print(save_path)
    draw_box(data,save_path)

def vis_cutoff():
    '''论文插图，cutoff对PN/ASR/ACC的影响'''
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = 'WaNet'
    random_seed = 1
    device = torch.device("cuda:1")
    CutOff_list,PN_rate_list,ASR_list,ACC_list = load_cutoff_data(dataset_name,model_name,attack_name,random_seed, device)
    save_dir = os.path.join("imgs", "cutoff")
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f"{attack_name}.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_cutoff(CutOff_list,PN_rate_list,ASR_list,ACC_list, save_path)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    vis_discussion_1()
    # vis_cutoff()