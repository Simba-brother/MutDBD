import sys
sys.path.append("./")
import os
import torch
from tqdm import tqdm
from codes.modelMutat import ModelMutat_2
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractTargetClassDataset

from codes.utils import create_dir
from codes import config




mutation_rate_list = config.mutation_rate_list
exp_root_dir = config.exp_root_dir
dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
from codes.datasets.cifar10.attacks.WaNet.ResNet18.attack import get_dict_state
dict_state = get_dict_state()
backdoor_model = dict_state["backdoor_model"]

mutation_num = 50

def gf_mutate():
    mutation_operator_name = "gf"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model =backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._gf_mut(scale=5)
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("gf_mutate() success")

def inverse_mutate():
    mutation_operator_name = "neuron_activation_inverse"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model =backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._neuron_activation_inverse()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("inverse_mutate() success")

def block_mutate():
    mutation_operator_name = "neuron_block"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model =backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._neuron_block()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("block_mutate() success")

def switch_mutate():
    mutation_operator_name = "neuron_switch"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model =backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._neuron_switch()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("switch_mutate() success")

def weight_shuffle_mutate():
    mutation_operator_name = "weight_shuffle"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model =backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._weight_shuffling()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("weight_shuffle_mutate() success")

def combination_mutate():
    gf_mutate()
    inverse_mutate()
    block_mutate()
    switch_mutate()
    weight_shuffle_mutate()


'''
评估变异后的模型
'''
# 获得Target class中的clean set 和poisoned set


if __name__ == "__main__":
    # combination_mutate()
    pass