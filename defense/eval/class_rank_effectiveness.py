import torch
import os
import numpy as np
from utils.common_utils import read_yaml
from datasets.posisoned_dataset import get_all_dataset
from mid_data_loader import get_class_rank, get_our_method_state
from models.model_loader import get_model
from defense.our.sample_select import sort_sample_id
from torch.utils.data import DataLoader,Subset
from utils.calcu_utils import compare_WTL
from utils.dataset_utils import split_method

def class_rank_analyse():
    '''class  rank 有效性分析'''
    # 后门信息
    backdoor_data = torch.load(os.path.join(exp_root_dir, "ATTACK",
                            dataset_name, model_name, attack_name,
                            "backdoor_data.pth"), map_location="cpu")
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 数据集
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset( dataset_name, model_name, attack_name,poisoned_ids)
    class_rank = get_class_rank(dataset_name, model_name, attack_name)
    no_class_list = []
    class_list = []
    for r_seed in range(1,11):
        defensed_state_dict_path, selected_state_dict_path = get_our_method_state(dataset_name, model_name, attack_name, 1)
        select_model = get_model(dataset_name,model_name)
        select_model.load_state_dict(torch.load(selected_state_dict_path,map_location="cpu"))
        # 中毒样本切分结果
        no_class_rank_p_num, choiced_num, poisoning_rate = split_method(
            select_model,
            poisoned_trainset,
            poisoned_ids,
            device,
            class_rank = None,
            choice_rate = 0.6
        )
        class_rank_p_num, class_rank_choiced_num, class_rank_poisoning_rate = split_method(
            select_model,
            poisoned_trainset,
            poisoned_ids,
            device,
            class_rank = class_rank,
            choice_rate = 0.6
        )

        no_class_list.append(no_class_rank_p_num)
        class_list.append(class_rank_p_num)
    no_class_avg = round(np.mean(no_class_list),3)
    class_avg = round(np.mean(class_list),3)
    wtl_res = compare_WTL(class_list,no_class_list,"small","mannwhitneyu")
    return class_avg, no_class_avg, wtl_res

if __name__ == "__main__":
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    device = torch.device("cuda:0")
    config = read_yaml("config.yaml")
    exp_root_dir = config["exp_root_dir"]


    # class_rank_list = []
    # no_class_rank_list = []
    # for dataset_name in ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
    #     class_num = get_classNum(dataset_name)
    #     for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #         if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #             continue
    #         for attack_name in ["BadNets","IAD","Refool", "WaNet"]:
    #             print(f"{dataset_name}|{model_name}|{attack_name}")
    #             no_class_rank_p_num,class_rank_p_num = class_rank_analyse()
    #             class_rank_list.append(class_rank_p_num)
    #             no_class_rank_list.append(no_class_rank_p_num)
    # class_rank_count = 0
    # no_class_rank_count = 0
    # for scence_idx in range(len(class_rank_list)):
    #     class_rank_pm =  class_rank_list[scence_idx]
    #     no_class_rank_pm =  no_class_rank_list[scence_idx]
    #     min_v =  min(class_rank_pm,no_class_rank_pm)
    #     if class_rank_pm == min_v:
    #         class_rank_count += 1
    #     if no_class_rank_count == min_v:
    #         no_class_rank_count += 1
    
    # avg_class_rank = round(np.mean(class_rank_list),3)
    # avg_no_class_rank = round(np.mean(no_class_rank_list),3)
    # wtl_res = compare_WTL(class_rank_list,no_class_rank_list,"small","wilcoxon")

    # print("avg_class_rank:",avg_class_rank)
    # print("avg_no_class_rank:",avg_no_class_rank)
    # print("wtl_res:",wtl_res)
    # print("class_rank_count:",class_rank_count)
    # print("no_class_rank_count:",no_class_rank_count)



    '''class rank 有效性分析'''
    # device = torch.device("cuda:1")
    # for dataset_name in ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
    #     class_num = get_classNum(dataset_name)
    #     for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #         if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #             continue
    #         for attack_name in ["BadNets","IAD","Refool", "WaNet"]:
    #             print(f"{dataset_name}|{model_name}|{attack_name}")
    #             class_avg, no_class_avg, wtl_res = class_rank_analyse() 
    #             print(f"class_AVG:{class_avg},no_class_AVG:{no_class_avg},wtl:{wtl_res}")