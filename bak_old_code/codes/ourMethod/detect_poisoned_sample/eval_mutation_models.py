
from codes import config
import setproctitle
import os
import logging
import torch
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
import joblib

def get_mutationModelPredLabels(dataset,ratio):
    mutations_dir = os.path.join(
        config.exp_root_dir,
        "MutationModels",
        config.dataset_name,
        config.model_name,
        config.attack_name
        )
    device = torch.device(f"cuda:{config.gpu_id}")
    ans={}
    for operator in config.mutation_name_list:
        ans[operator] = []
        for i in range(config.mutation_model_num):
            mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
            backdoor_model.load_state_dict(torch.load(mutation_model_path))
            em = EvalModel(backdoor_model,dataset,device)
            pred_labels = em.get_pred_labels() 
            ans[operator].append(pred_labels)
    return ans


if __name__ == "__main__":
    # 进程名称
    proctitle = f"EvalMutaionModelsPredLabelsOnSuspiciousClasses|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cuda:{config.gpu_id}")

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "EvalMutaionModelsPredLabelsOnSuspiciousClasses.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    # 加载后门模型数据
    backdoor_data_path = os.path.join(config.exp_root_dir, "ATTACK", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_ids =backdoor_data["poisoned_ids"]
    # 加载后门模型中的可疑classes
    suspicious_classes_dict = joblib.load(os.path.join(
        config.exp_root_dir,
        "SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "SuspiciousClasses_SK_Precision.data"
    ))
    
    for ratio in config.fine_mutation_rate_list:
        # 获得该变异率下suspicious_classes
        suspicious_classes = suspicious_classes_dict[ratio]
        # 从poisoned_trainset抽取出suspicious_classes dataset
        suspiciousClassesDataset = ExtractSuspiciousClassesDataset(poisoned_trainset,suspicious_classes,poisoned_ids)
        pred_label_dict =  get_mutationModelPredLabels(suspiciousClassesDataset,ratio)
        # 保存结果
        save_dir = os.path.join(
            config.exp_root_dir,
            "SuspiciousClassesPredLabel",
            config.dataset_name, 
            config.model_name, 
            config.attack_name,
            str(ratio)
        )
        os.makedirs(save_dir,exist_ok=True)
        save_file_name = "res.data"
        save_file_path = os.path.join(save_dir,save_file_name)
        joblib.dump(pred_label_dict,save_file_path)
        logging.debug(f"变异率:{str (ratio)},变异模型在suspiciousClasses dataset的预测标签结果保存在:{save_file_path}")




