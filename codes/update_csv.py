'''
重要：
变异模型评估过后生成EvalMutationToCSV目录下preLabel.csv中缺乏original backdoor model的预测标签，这里给补上。
'''
import os
from codes import config
import torch
import pandas as pd
from codes.common.eval_model import EvalModel
import logging
import setproctitle
# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_poisoned_dataset as gtsrb_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_poisoned_dataset as gtsrb_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_poisoned_dataset as gtsrb_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_poisoned_dataset as gtsrb_WaNet_gen_poisoned_dataset

# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset as imagenet_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_poisoned_dataset as imagenet_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_poisoned_dataset as imagenet_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_poisoned_dataset as imagenet_WaNet_gen_poisoned_dataset


def get_poisoned_trainset(poisoned_ids):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(model_name,poisoned_ids,"train")
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
    return poisoned_trainset


if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    mutation_rate_list  = [0.03,0.05,0.07,0.09,0.1]
    gpu_id = 0
    # 进程名称
    exp_name = "UpdateCSV"
    proctitle = f"{exp_name}|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log","UpdateCSV",dataset_name,model_name,attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = f"{exp_name}.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    try:
        # 加载后门模型数据
        backdoor_data_path = os.path.join(
            exp_root_dir, 
            "ATTACK",  
            dataset_name, 
            model_name, 
            attack_name, 
            "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        backdoor_model = backdoor_data["backdoor_model"]
        poisoned_ids = backdoor_data["poisoned_ids"]
        poisoned_trainset = get_poisoned_trainset(poisoned_ids)
        device = torch.device(f"cuda:{gpu_id}")
        e = EvalModel(backdoor_model,poisoned_trainset,device)
        original_backdoorModel_preLabel_list = e.get_pred_labels()
        for rate in mutation_rate_list:
            csv_dir = os.path.join(
                exp_root_dir,
                "EvalMutationToCSV",
                dataset_name,
                model_name,
                attack_name,
                str(rate)
            )
            csv_name = "preLabel.csv"
            csv_path = os.path.join(csv_dir,csv_name)
            df = pd.read_csv(csv_path)
            # 添加一列
            df["original_backdoorModel_preLabel"] = original_backdoorModel_preLabel_list
            df.to_csv(csv_path,index=False)
            logging.debug(csv_path)
        logging.debug("End")
    except Exception as e:
        logging.error("发生异常:%s",e)