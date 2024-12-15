'''
生成变异模型
'''
from tqdm import tqdm
from codes import config
import os
import torch
from codes.ourMethod.model_mutation.mutationOperator import MutaionOperator
from codes.scripts.dataset_constructor import *


class OpType(object):
    GF  = 'GF' # Gaussian Fuzzing
    WS = 'WS' # Weight Shuffling
    NAI = 'NAI' # Neuron Activation Inverse
    NB = 'NB' # Neuron Block
    NS = 'NS' # Neuron Switch

def gen_mutation_models(model,save_dir,op_type):
    for ration in config.fine_mutation_rate_list:
        for i in range(config.mutation_model_num):
            mo = MutaionOperator(
                ration, 
                model, 
                verbose=False,
                device=device
                )
            if op_type == OpType.GF:
                mutated_model = mo.gf()
                temp_save_dir = os.path.join(save_dir,str(ration),"Gaussian_Fuzzing")
                
            elif op_type == OpType.WS:
                mutated_model = mo.ws()
                temp_save_dir = os.path.join(save_dir,str(ration),"Weight_Shuffling")
            elif op_type == OpType.NAI:
                mutated_model = mo.nai()
                temp_save_dir = os.path.join(save_dir,str(ration),"Neuron_Activation_Inverse")
            elif op_type == OpType.NB:
                mutated_model = mo.nb()
                temp_save_dir = os.path.join(save_dir,str(ration),"Neuron_Block")
            elif op_type == OpType.NS:
                mutated_model = mo.ns()
                temp_save_dir = os.path.join(save_dir,str(ration),"Neuron_Switch")
            save_file_name = f"model_{i}"
            os.makedirs(temp_save_dir,exist_ok=True)
            save_path = os.path.join(temp_save_dir,save_file_name)
            torch.save(mutated_model.state_dict(),save_path)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    save_dir = os.path.join(
        config.exp_root_dir,
        "mutation_models",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        )
    os.makedirs(save_dir,exist_ok=True)

    # 获得backdoor_data
    backdoor_data_path = os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    # poisoned_trainset =backdoor_data["poisoned_trainset"]
    # poisoned_testset =backdoor_data["poisoned_testset"]
    # poisoned_ids =backdoor_data["poisoned_ids"]
    # clean_testset =backdoor_data["clean_testset"]
    # # 数据预transform,为了后面训练加载的更快
    # poisoned_trainset = ExtractDataset(poisoned_trainset)
    # pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset,poisoned_ids)
    # purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset,poisoned_ids)
    # poisoned_testset = ExtractDataset(poisoned_testset)
    # victim model
    for op in tqdm([OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]):
        gen_mutation_models(backdoor_model,save_dir,op)



