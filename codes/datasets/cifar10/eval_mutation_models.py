import sys
sys.path.append("./")
import os
import torch
from tqdm import tqdm
from codes.modelMutat import ModelMutat_2
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractTargetClassDataset

from codes.utils import create_dir
from codes import config
from codes.eval_model import EvalModel
from codes import draw


mutation_rate_list = config.mutation_rate_list
exp_root_dir = config.exp_root_dir
dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
from codes.datasets.cifar10.attacks.Blended.ResNet18.attack import get_dict_state
dict_state = get_dict_state()
backdoor_model = dict_state["backdoor_model"]

mutation_num = 50
target_class_idx = 1
target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], target_class_idx)
target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], target_class_idx)

y_dict = {"poisoned":[], "clean":[]}
for mutation_rate in mutation_rate_list:
    mutation_model_weight_dir_path = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, str(mutation_rate))
    p_acc_list = []
    c_acc_list = []
    for m_i in range(mutation_num):
        weight_file_name = f"mutated_model_{m_i}.pth"
        weight_path = os.path.join(mutation_model_weight_dir_path, weight_file_name)
        backdoor_model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        e = EvalModel(backdoor_model, target_class_poisoned_set)
        p_acc = e._eval_acc()
        e = EvalModel(backdoor_model, target_class_clean_set)
        c_acc = e._eval_acc()
        p_acc_list.append(p_acc)
        c_acc_list.append(c_acc)
    mean_p_acc = sum(p_acc_list)/len(p_acc_list)
    mean_c_acc = sum(c_acc_list)/len(c_acc_list)
    y_dict["poisoned"].append(mean_p_acc)
    y_dict["clean"].append(mean_c_acc)

