'''
检测target class
'''
import os
from codes import config
import torch
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
import joblib
import logging
import setproctitle

def get_mutated_models_eval_report(dataset):
    ans = {}
    # 加载变异模型权重
    mutations_dir = os.path.join(
        config.exp_root_dir,
        "MutationModels",
        config.dataset_name,
        config.model_name,
        config.attack_name
        )
    ans = {}
    for ratio in config.fine_mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        ans[ratio] = {}
        for operator in config.mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            ans[ratio][operator] = []
            for i in range(config.mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device)
                report = em.eval_classification_report()
                ans[ratio][operator].append(report)
    return ans

if __name__ == "__main__":
    # 进程名称
    exp_name = "EvalMutationResult_for_SuspiciousClasses"
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

    # 实验主体代码
    try:
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
        # 开始评估变异模型
        eval_report = get_mutated_models_eval_report(poisoned_trainset)
        # 保存结果
        save_dir = os.path.join(
            config.exp_root_dir,
            exp_name,
            config.dataset_name, 
            config.model_name, 
            config.attack_name
        )
        os.makedirs(save_dir,exist_ok=True)
        save_file_name = "res.data"
        save_file_path = os.path.join(save_dir,save_file_name)
        joblib.dump(eval_report,save_file_path)
        logging.debug(f"评估变异模型结果保存在:{save_file_path}")
    except Exception as e:
        logging.error("发生异常:%s",e)
