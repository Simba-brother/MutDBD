'''
生成变异模型脚本
'''
import os
import torch
from tqdm import tqdm
from codes.ourMethod.modelMutat import ModelMutat_2
from codes.utils import create_dir
from collections import defaultdict



def gf_mutate(
        model,
        mutation_rate_list,
        mutation_num):
    '''
    Fun:
        高斯模糊变异
    Args:
        model: 待变异模型
        mutation_rate_list: 变异率list
        mutation_num: 每个变异率生成的变异模型的个数
    Return:
        res:{
            mutation_rate:[mutated_model_weight,..,]
        }

    '''
    res = defaultdict(list)
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model=model, mutation_rate=mutation_rate)
            mutated_model = mm._gf_mut(scale=5)
            res[mutation_rate].append(mutated_model.state_dict())
    return res

def weight_gf(backdoor_model):
    sp_mutation_rate_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
    sp_mutation_num = 50
    mutation_operator_name = "weight_gf"
    for mutation_rate in tqdm(sp_mutation_rate_list):
        for m_i in range(sp_mutation_num):
            mm = ModelMutat_2(original_model=backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._weight_gf()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("weight_gf() success")

def inverse_mutate(backdoor_model):
    mutation_operator_name = "neuron_activation_inverse"
    for mutation_rate in tqdm(mutation_rate_list):
        for m_i in range(mutation_num):
            mm = ModelMutat_2(original_model=backdoor_model, mutation_rate=mutation_rate)
            mutated_model = mm._neuron_activation_inverse()
            save_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
            create_dir(save_dir)
            save_file_name = f"mutated_model_{m_i}.pth"
            save_path = os.path.join(save_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
    print("inverse_mutate() success")

def block_mutate(backdoor_model):
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

def switch_mutate(backdoor_model):
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

def weight_shuffle_mutate(backdoor_model):
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

def mutate(model):
    gf_mutate(model, mutation_rate_list, mutation_num)
    inverse_mutate(backdoor_model)
    block_mutate(backdoor_model)
    switch_mutate(backdoor_model)
    weight_shuffle_mutate()
    print("combination_mutate() End")

if __name__ == "__main__":
    # start_time = time.perf_counter()
    # proctitle = dataset_name+"_"+attack_name+"_"+model_name+"_mutation"
    # print(proctitle)
    # setproctitle.setproctitle(proctitle)
    # mutate()
    # end_time = time.perf_counter()
    # execution_time = end_time -start_time
    # print(f"Execution time: {execution_time} seconds")
    pass