'''
重要
生成变异模型py文件
'''
from tqdm import tqdm
import os
import torch
import logging
import setproctitle
from codes.ourMethod.model_mutation.mutationOperator import MutaionOperator
from codes.scripts.dataset_constructor import *

class OpType(object):
    GF  = 'GF' # Gaussian Fuzzing
    WS = 'WS' # Weight Shuffling
    NAI = 'NAI' # Neuron Activation Inverse
    NB = 'NB' # Neuron Block
    NS = 'NS' # Neuron Switch

def gen_mutation_models(model,save_dir,op_type): 
    for ration in rate_list:
        for i in range(mutated_model_num):
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

def get_classNum(dataset_name):
    class_num = None
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    return class_num

if __name__ == "__main__":
    # 进程名称
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "ImageNet2012_subset" # ImageNet2012_subset
    model_name = "ResNet18"
    attack_name = "BadNets"
    class_num = get_classNum(dataset_name)
    # 变异率列表
    rate_list = [0.03,0.05,0.07,0.09,0.1] # [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # [0.03,0.05,0.07,0.09,0.1]
    # 每个变异算子在每个变异率下生成的变异模型数量
    mutated_model_num = 50
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
    LOG_FILE_NAME = "Mutations.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    try:
        # 获得backdoor_data
        backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name,attack_name, "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        backdoor_model = backdoor_data["backdoor_model"]
        for op in tqdm([OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]):
            logging.debug(op)
            gen_mutation_models(backdoor_model,save_dir,op)
        logging.debug("End")
    except Exception as e:
        logging.error("发生异常:%s",e)
    
    

