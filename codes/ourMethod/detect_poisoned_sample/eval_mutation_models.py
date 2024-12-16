
from codes import config
import setproctitle
import os
import logging
import torch
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
import joblib

def get_mutationModelPredLabels(dataset):
    mutations_dir = os.path.join(
        config.exp_root_dir,
        "mutation_models",
        config.dataset_name,
        config.model_name,
        config.attack_name
        )
    device = torch.device("cuda:0")
    ans = {}
    for ratio in config.fine_mutation_rate_list:
        ans[ratio] = {}
        for operator in config.mutation_name_list:
            ans[ratio][operator] = []
            for i in range(config.mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device)
                pred_labels = em.get_pred_labels() 
                ans[ratio][operator].append(pred_labels)
    return ans


if __name__ == "__main__":
    # 进程名称
    proctitle = f"EvalMutaionModelsPredLabelsOnTargeClass|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device("cuda:0")

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "EvalMutaionModelsPredLabelsOnTargeClass.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    # 加载后门模型数据
    backdoor_data_path = os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_ids =backdoor_data["poisoned_ids"]
    target_class = 1
    clean_targetClassDataset = ExtractCleanTargetClassDataset(poisoned_trainset,target_class,poisoned_ids)
    poisoned_targetClassDataset = ExtractCleanTargetClassDataset(poisoned_trainset,target_class,poisoned_ids)

    condidate_dataset = CombinDataset(poisoned_targetClassDataset,clean_targetClassDataset)
    gt_isPoisoned = []
    for _ in len(poisoned_targetClassDataset):
        gt_isPoisoned.append(True) # poisoned
    for _ in len(clean_targetClassDataset):
        gt_isPoisoned.append(False) # poisoned

    pred_label_ans =  get_mutationModelPredLabels(condidate_dataset)
    # 保存结果
    save_dir = os.path.join(
        config.exp_root_dir,
        "EvalMutationResult",
        config.dataset_name, 
        config.model_name, 
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)

    save_file_name = "pred_label_ans.data"
    save_file_path = os.path.join(save_dir,save_file_name)
    joblib.dump(pred_label_ans,save_file_path)
    logging.debug(f"评估变异模型在候选数据的预测标签结果保存在:{save_file_path}")

    save_file_name = "gt_isPoisoned.data"
    save_file_path = os.path.join(save_dir,save_file_name)
    joblib.dump(gt_isPoisoned,save_file_path)
    logging.debug(f"评估变异模型在候选数据是否是中毒样本groundTruth结果保存在:{save_file_path}")




