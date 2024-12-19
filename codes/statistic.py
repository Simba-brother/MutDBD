import os
import torch
import config
from codes.scripts.dataset_constructor import *

exp_root_dir = config.exp_root_dir
dataset_name = config.dataset_name
model_name =  config.model_name
attack_name = config.attack_name
target_class_idx = config.target_class_idx

backdoor_data_path = os.path.join(exp_root_dir,"ATTACK",dataset_name,model_name,attack_name,"backdoor_data.pth")
backdoor_data = torch.load(backdoor_data_path, map_location="cpu")

poisoned_trainset = backdoor_data["poisoned_trainset"]
poisoned_ids = backdoor_data["poisoned_ids"]

targetClass_trainset = ExtractTargetClassDataset(poisoned_trainset, target_class_idx)
purePoisoned_trainset = ExtractDatasetByIds(poisoned_trainset,poisoned_ids)

print(f"target class样本数量:{len(targetClass_trainset)}")
rate = round(len(purePoisoned_trainset)/len(targetClass_trainset),3)
print(f"攻击类别中木马样本比例:{rate}")





