'''
生成变异模型
'''
from tqdm import tqdm
from codes import config
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
                mutated_model = mo.ns(skip=config.class_num)
                temp_save_dir = os.path.join(save_dir,str(ration),"Neuron_Switch")
            save_file_name = f"model_{i}.pth"
            os.makedirs(temp_save_dir,exist_ok=True)
            save_path = os.path.join(temp_save_dir,save_file_name)
            # 保存变异模型的状态字典
            torch.save(mutated_model.state_dict(),save_path)

if __name__ == "__main__":
    # 进程名称
    proctitle = f"Mutations|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cpu")
    # 变异模型保存目录
    save_dir = os.path.join(
        config.exp_root_dir,
        "MutationModels",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        )
    # os.makedirs(save_dir,exist_ok=True)
    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "Mutations.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    try:
        # 获得backdoor_data
        backdoor_data_path = os.path.join(config.exp_root_dir, "ATTACK", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        backdoor_model = backdoor_data["backdoor_model"]
        # [OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]
        for op in tqdm([OpType.GF,OpType.WS,OpType.NAI,OpType.NB,OpType.NS]):
            logging.debug(op)
            gen_mutation_models(backdoor_model,save_dir,op)
        logging.debug("End")
    except Exception as e:
        logging.error("发生异常:%s",e)
    
    

