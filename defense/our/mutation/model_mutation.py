'''
基于Backdoor model生成变异模型
'''
import time
from tqdm import tqdm
import os
import torch
import logging
import setproctitle
from defense.our.mutation.mutation_operator import MutaionOperator
from utils.common_utils import read_yaml
from utils.dataset_utils import get_class_num
from models.model_loader import get_model
from utils.common_utils import convert_to_hms

class OpType(object):
    GF  = 'GF' # Gaussian Fuzzing
    WS = 'WS' # Weight Shuffling
    NAI = 'NAI' # Neuron Activation Inverse
    NB = 'NB' # Neuron Block
    NS = 'NS' # Neuron Switch

def gen_mutation_models(model,save_dir,op_type, model_id_list): 
    for ration in rate_list:
        for i in model_id_list:
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
                mutated_model = mo.ns(skip=class_num)
                temp_save_dir = os.path.join(save_dir,str(ration),"Neuron_Switch")
            save_file_name = f"model_{i}.pth"
            os.makedirs(temp_save_dir,exist_ok=True)
            save_path = os.path.join(temp_save_dir,save_file_name)
            # 保存变异模型的状态字典
            torch.save(mutated_model.state_dict(),save_path)

if __name__ == "__main__":
    # one-scence
    '''
    # 进程名称
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10" # CIFAR10 GTSRB, ImageNet2012_subset
    model_name = "VGG19" # ResNet18,VGG19,DenseNet
    attack_name = "Refool" # BadNets,IAD,Refool,WaNet, LabelConsistent
    class_num = get_class_num(dataset_name) # 该数据集的class nums
    # 变异率列表
    rate_list = [0.001] # [0.01,0.03,0.05,0.07,0.09,0.1] # [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # [0.03,0.05,0.07,0.09,0.1]
    # 每个变异算子在每个变异率下生成的变异模型数量
    model_id_list = list(range(100))
    proctitle = f"Mutations|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    # 变异模型的生成使用cpu设备即可
    device = torch.device(f"cpu")
    # 变异模型保存目录
    save_dir = os.path.join(
        exp_root_dir,
        "MutationModels", # "MutationsForDiscussion",
        dataset_name,
        model_name,
        attack_name,
        )
    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log","Mutations",dataset_name,model_name,attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "Mutations_0.1%.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    try:
        # 获得backdoor_data
        backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name,attack_name, "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        
        if "backdoor_model" in backdoor_data.keys():
            backdoor_model = backdoor_data["backdoor_model"]
        else:
            model = get_model(dataset_name, model_name)
            state_dict = backdoor_data["backdoor_model_weights"]
            model.load_state_dict(state_dict)
            backdoor_model = model

        for op in tqdm([OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]):
            logging.debug(op)
            gen_mutation_models(backdoor_model,save_dir,op, model_id_list)
        logging.debug("End")
    except Exception as e:
        logging.error("发生异常:%s",e)
    '''
    # all-scence
    # 进程名称
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    rate_list = [0.007] # [0.01,0.03,0.05,0.07,0.09,0.1] # [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # [0.03,0.05,0.07,0.09,0.1]
    # 每个变异算子在每个变异率下生成的变异模型数量
    model_id_list = list(range(100))
    # 变异模型的生成使用cpu设备即可
    device = torch.device(f"cpu")
    r_seed_list = [1]

    exp_sence_params =  {
        "exp_root_dir":exp_root_dir,
        "dataset_name_list":dataset_name_list,
        "model_name_list":model_name_list,
        "attack_name_list":attack_name_list,
        "rate_list":rate_list,
        "r_seed_list":r_seed_list
    }
    print(exp_sence_params)

    all_scence_start_time = time.perf_counter()
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                class_num = get_class_num(dataset_name) # 该数据集的class nums
                # 变异模型保存目录
                save_dir = os.path.join(
                    exp_root_dir,
                    "MutationModels", # "MutationsForDiscussion",
                    dataset_name,
                    model_name,
                    attack_name,
                    )
                # 日志保存目录
                # LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
                # LOG_FILE_DIR = os.path.join("log","Mutations",dataset_name,model_name,attack_name)
                # os.makedirs(LOG_FILE_DIR,exist_ok=True)
                # LOG_FILE_NAME = f"Mutations_{rate_list[0]*100}%.log"
                # LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
                # logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
                # logging.debug(f"Mutations|{dataset_name}|{model_name}|{attack_name}")
                print(f"Mutations|{dataset_name}|{model_name}|{attack_name}")
                print(f"mutation_rate_list: {rate_list}")
                print(f"save_dir:{save_dir}")
                one_scence_start_time = time.perf_counter()
                try:
                    # 获得backdoor_data
                    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name,attack_name, "backdoor_data.pth")
                    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
                    
                    if "backdoor_model" in backdoor_data.keys():
                        backdoor_model = backdoor_data["backdoor_model"]
                    else:
                        model = get_model(dataset_name, model_name)
                        state_dict = backdoor_data["backdoor_model_weights"]
                        model.load_state_dict(state_dict)
                        backdoor_model = model

                    for op in tqdm([OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]):
                        print(op)
                        gen_mutation_models(backdoor_model,save_dir,op,model_id_list)
                    print("End")
                except Exception as e:
                    logging.error("发生异常:%s",e)
                one_scence_end_time = time.perf_counter()
                one_scence_cost_time = one_scence_end_time - one_scence_start_time
                hours, minutes, seconds = convert_to_hms(one_scence_cost_time)
                print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
    all_scence_end_time = time.perf_counter()
    all_scence_cost_time = all_scence_end_time - all_scence_start_time
    hours, minutes, seconds = convert_to_hms(all_scence_cost_time)
    print(f"all-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")


