'''
使用变异模型对训练集进行评估
'''
import os

import torch
import logging
import setproctitle
import pandas as pd

from codes import config
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel


def get_mutation_models_pred_labels(dataset):
    mutations_dir = os.path.join(
        config.exp_root_dir,
        "MutationModels",
        config.dataset_name,
        config.model_name,
        config.attack_name
    )
    eval_ans = {}
    for ratio in config.fine_mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in config.mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(config.mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device)
                pred_label_list = em.get_pred_labels()
                eval_ans[ratio][operator].append(pred_label_list)
    return eval_ans

def get_mutation_models_confidence(dataset):
    mutations_dir = os.path.join(
        config.exp_root_dir,
        "MutationModels",
        config.dataset_name,
        config.model_name,
        config.attack_name
    )
    eval_ans = {}
    for ratio in config.fine_mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in config.mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(config.mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device)
                confidence_list = em.get_confidence_list()
                eval_ans[ratio][operator].append(confidence_list)
    return eval_ans

def ansToCSV(data_dict,save_path):

    # 所有的变异模型列
    # global_model_id：[0-99]是GF变异模型，[100-199]是WS变异模型，
    # [200-299]是NAI变异模型，[300-399]是NB变异模型，[400-499]是NS变异模型
    total_dict = {}
    global_model_id = 0
    for operator in config.mutation_name_list:
        for i in range(config.mutation_model_num):
            pred_label_list = data_dict[operator][i]
            model_name = f"model_{global_model_id}"
            total_dict[model_name] = pred_label_list # 变异模型列
            global_model_id += 1

    # sampled_id列,GT_label列和isPoisoned列
    total_dict["sampled_id"] = []
    total_dict["GT_label"] = []
    total_dict["isPoisoned"] = []
    for i in range(len(poisoned_trainset)):
        total_dict["sampled_id"].append(i)
        total_dict["GT_label"].append(poisoned_trainset[i][1])
        if i in poisoned_ids:
            total_dict["isPoisoned"].append(True)
        else:
            total_dict["isPoisoned"].append(False)

    # 构建DataFrame
    df = pd.DataFrame(total_dict)
    # 保存为csv
    df.to_csv(save_path,index=False)




if __name__ == "__main__":
    # 进程名称
    exp_name = "EvalMutationToCSV_confidence" 
    proctitle = f"{exp_name}|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cuda:{config.gpu_id}")
    

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = f"{exp_name}.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    try:
        logging.debug(f"开始:加载后门模型数据")
        backdoor_data_path = os.path.join(
            config.exp_root_dir, 
            "ATTACK", 
            config.dataset_name, 
            config.model_name, 
            config.attack_name, 
            "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        backdoor_model = backdoor_data["backdoor_model"]
        poisoned_trainset = backdoor_data["poisoned_trainset"]
        poisoned_ids = backdoor_data["poisoned_ids"]
        logging.debug(f"开始:得到所有变异模型在poisoned trainset上的预测标签结果")
        # mutation_models_pred_labels_dict = get_mutation_models_pred_labels(poisoned_trainset)
        mutation_models_confidence_dict = get_mutation_models_confidence(poisoned_trainset)
        logging.debug(f"开始:将结果整理为csv文件")
        for rate in [0.05]:
            data_dict = mutation_models_confidence_dict[rate]
            save_dir = os.path.join(
                config.exp_root_dir,
                exp_name,
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate)
            )
            os.makedirs(save_dir,exist_ok=True)
            save_file_name = "confidence.csv" # preLabel.csv or prob.csv
            save_file_path = os.path.join(save_dir,save_file_name)
            ansToCSV(data_dict,save_file_path)
            logging.debug(f"csv保存在:{save_file_path}")
    except Exception as e:
        logging.error("发生异常:%s",e)





