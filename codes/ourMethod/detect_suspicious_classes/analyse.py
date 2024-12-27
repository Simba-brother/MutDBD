import os
import joblib
import logging
import setproctitle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report,confusion_matrix
from collections import defaultdict
from codes import config
from codes.common.eval_model import EvalModel
# 自定义数据集构建包
from codes.scripts.dataset_constructor import *


def get_mutated_models_eval_report(model,dataset):
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
                model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(model,dataset,device)
                report = em.eval_classification_report()
                ans[ratio][operator].append(report)
    return ans


def main_v1():
    # 加载后门数据
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
    poisoned_ids = backdoor_data["poisoned_ids"]
    cleanDatasetOfTargetClass = ExtractCleanDatasetOfTargetClass(poisoned_trainset, config.target_class_idx, poisoned_ids)
    poisonedDataset = ExtractDatasetByIds(poisoned_trainset,poisoned_ids)

    clean_report = get_mutated_models_eval_report(backdoor_model,cleanDatasetOfTargetClass)
    poisoned_report = get_mutated_models_eval_report(backdoor_model,poisonedDataset)

    save_dir = os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
    )
    os.makedirs(save_dir)
    save_file_name = "clean_report.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(clean_report,save_path)
    logging.debug(f"clean_report save_path:{save_path}")

    save_file_name = "poisoned_report.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(poisoned_report,save_path)
    logging.debug(f"poisoned_report save_path:{save_path}")


def main_v2():
    res = {}
    for rate in config.fine_mutation_rate_list:
        res[rate] = defaultdict(list)
        logging.debug(f"变异率:{rate}")
        df = pd.read_csv(os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "data.csv"))
        target_class_clean_df = df.loc[(df["GT_label"] == config.target_class_idx) & df["isPoisoned"] == False]
        target_class_poisoned_df = df.loc[(df["GT_label"] == config.target_class_idx) & df["isPoisoned"] == True]

        GT_label_list = df["GT_label"]
        for mutated_model_global_id in range(500):
            model_col_name = f"model_{mutated_model_global_id}"
            pred_label_clean_list = list(target_class_clean_df[model_col_name])
            pred_label_poisoned_list = list(target_class_poisoned_df[model_col_name])
            clean_report = classification_report(GT_label_list,pred_label_clean_list,output_dict=True)
            poisoned_report = classification_report(GT_label_list,pred_label_poisoned_list,output_dict=True)
            clean_recall = clean_report[str(config.target_class_idx)]["recall"]
            poisoned_recall = poisoned_report[str(config.target_class_idx)]["recall"]
            res[rate]["clean"].append(clean_recall)
            res[rate]["poisoned"].append(poisoned_recall)
    
    save_dir = os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
    )
    os.makedirs(save_dir)
    save_file_name = "res.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(res,save_path)
    logging.debug(f"save_path:{save_path}")

if __name__ == "__main__":
    # 进程名称
    exp_name = "TargetClassAnalyse"
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
        # main_v1()
        # main_v2()
        pass
    except Exception as e:
        logging.error("发生异常: %s", e)
