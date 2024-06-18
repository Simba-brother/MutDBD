import os
import torch
import config
from codes.scripts.dataset_constructor import (ExtractDataset, 
                                        PureCleanTrainDataset, 
                                        PurePoisonedTrainDataset, 
                                        ExtractTargetClassDataset, 
                                        ExtractDatasetByIds, CombinDataset)
dataset_name = config.dataset_name
model_name =  config.model_name
attack_name = config.attack_name
exp_root_dir = config.exp_root_dir
dict_state_path = os.path.join(exp_root_dir,"attack",
                               dataset_name,model_name,attack_name,"attack","dict_state.pth")
dict_state = torch.load(dict_state_path, map_location="cpu")
poisoned_trainset = dict_state["poisoned_trainset"]
targetset_train = ExtractTargetClassDataset(poisoned_trainset, target_class_idx=1)
purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
pure_clean_targetset_train = ExtractTargetClassDataset(pureCleanTrainDataset, target_class_idx=1)
pure_poisoned_targetset_train =  ExtractTargetClassDataset(purePoisonedTrainDataset,target_class_idx=1)
assert len(pure_poisoned_targetset_train)+len(pure_clean_targetset_train) == len(targetset_train), "数量不对"
print(f"攻击类别中干净样本数量:{len(pure_clean_targetset_train)}")
print(f"攻击类别中木马样本数量:{len(pure_poisoned_targetset_train)}")
print(f"攻击类别总样本数量:{len(targetset_train)}")
rate = round(len(pure_poisoned_targetset_train)/len(targetset_train),4)
print(f"攻击类别中木马样本比例:{rate}")





