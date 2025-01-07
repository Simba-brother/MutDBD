import os
from codes import config
import torch
import pandas as pd
from codes.common.eval_model import EvalModel
import logging
import setproctitle

if __name__ == "__main__":
    # 进程名称
    exp_name = "UpdateCSV"
    proctitle = f"{exp_name}|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = f"{exp_name}.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    # 加载后门模型数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    device = torch.device(f"cuda:{config.gpu_id}")
    e = EvalModel(backdoor_model,poisoned_trainset,device)
    original_backdoorModel_preLabel_list = e.get_pred_labels()
    for rate in config.fine_mutation_rate_list:
        csv_dir = os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate)
        )
        csv_name = "preLabel.csv"
        csv_path = os.path.join(csv_dir,csv_name)
        df = pd.read_csv(csv_path)
        # 添加一列
        df["original_backdoorModel_preLabel"] = original_backdoorModel_preLabel_list
        df.to_csv(csv_path,index=False)
        logging.debug(csv_path)