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
import setproctitle
import joblib
from collections import defaultdict
from codes.scripts.baseData import BaseData

mutation_rate_list = config.mutation_rate_list
exp_root_dir = config.exp_root_dir
dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_operator_name_list = config.mutation_name_list

from codes.datasets.cifar10.attacks.WaNet.ResNet18.attack import get_dict_state
dict_state = get_dict_state()
backdoor_model = dict_state["backdoor_model"]

mutation_num = 50
target_class_idx = 1

device = torch.device("cuda:1")


def eval_mutated_model(mutation_operator_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在整个后门训练集上各个分类上的report
    '''
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = f"eval_poisoned_trainset_report.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 整个后门训练集
    poisoned_trainset = dict_state["poisoned_trainset"]

    report_data = defaultdict(list)
    for mutation_rate in tqdm(mutation_rate_list):
        mutation_models_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
        for m_i in range(mutation_num):
            state_dict = torch.load(os.path.join(mutation_models_dir, f"mutated_model_{m_i}.pth"), map_location="cpu")
            backdoor_model.load_state_dict(state_dict)
            evalModel = EvalModel(backdoor_model, poisoned_trainset, device)
            report = evalModel._eval_classes_acc()
            report_data[mutation_rate].append(report)
    joblib.dump(report_data, save_path)
    return report_data

def eval_mutated_model_in_target_class(mutation_operator_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在后门训练集上target class上(clean,posioned,整体)的accuracy
    '''
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = f"eval_poisoned_trainset_target_class.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 目标类索引
    target_class_idx = 1
    # 把目标类数据集分为clean和poisoned
    target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], target_class_idx)
    target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], target_class_idx)
    whole_target_set = ExtractTargetClassDataset(dict_state["poisoned_trainset"], target_class_idx)
    
    # {mutation_ratio:[{"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole}]}
    res_dict = dict()
    for mutation_rate in tqdm(mutation_rate_list):
        mutation_models_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate)) 
        # 获得该变异率下的变异模型的state_dict
        temp_list = []
        for m_i in range(mutation_num):
            mutation_model_state_dict_path = os.path.join(mutation_models_dir, f"mutated_model_{m_i}.pth")
            mutation_model_state_dict = torch.load(mutation_model_state_dict_path, map_location="cpu")
            backdoor_model.load_state_dict(mutation_model_state_dict)
            e = EvalModel(backdoor_model, target_class_clean_set, device)
            acc_clean = e._eval_acc()
            e = EvalModel(backdoor_model, target_class_poisoned_set, device)
            acc_poisoned = e._eval_acc()
            e = EvalModel(backdoor_model, whole_target_set, device)
            acc_whole = e._eval_acc()
            temp_list.append({"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole})
        res_dict[mutation_rate] = temp_list
    joblib.dump(res_dict, save_path)
    print("eval_mutated_model_in_target_class() success")
    return res_dict



if __name__ == "__main__":
    setproctitle.setproctitle(attack_name+"_eval_mutated_models_on_targetClass")
    for mutation_operator in mutation_operator_name_list:
        print("mutation_operator:{mutation_operator}")
        eval_mutated_model_in_target_class(mutation_operator_name=mutation_operator)
