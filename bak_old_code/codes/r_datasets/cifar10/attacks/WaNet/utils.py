import os
import torch
from codes.scripts.dataset_constructor import ExtractDataset

def create_backdoor_data(attack_dict_path,save_path):
    
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_ids = poisoned_trainset.poisoned_set
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)

    backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_trainset":poisoned_trainset,
        "poisoned_testset":poisoned_testset,
        "clean_testset":clean_testset,
        "poisoned_ids":poisoned_ids
    }
    
    torch.save(backdoor_data,save_path)
    print(f"backdoor_data is saved in {save_path}")