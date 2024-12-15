
import os
import sys
import setproctitle
import torch
import time
import torch.nn as nn 
from codes import models
from codes import config
from codes.scripts.dataset_constructor import *
from codes.tools import model_train_test,EvalModel
from collections import defaultdict
import logging




if __name__ == "__main__":

    proctitle = f"Eval|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    print(f"proctitle:{proctitle}")
    # 获得backdoor_data
    backdoor_data_path = os.path.join(config.exp_root_dir, 
                                    "attack", 
                                    config.dataset_name, 
                                    config.model_name, 
                                    config.attack_name, 
                                    "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    # 后门模型
    backdoor_model = backdoor_data["backdoor_model"]
    # 投毒的训练集,带有一定的比例
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    # 投毒的测试集
    poisoned_testset =backdoor_data["poisoned_testset"]
    # 训练集中投毒的索引
    poisoned_ids =backdoor_data["poisoned_ids"]
    # 干净的测试集
    clean_testset =backdoor_data["clean_testset"]

    # 让数据在此处经过transform,为了后面训练加载的更快
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset,poisoned_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset,poisoned_ids)
    poisoned_testset = ExtractDataset(poisoned_testset)

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device)
    print("No defence ASR:",evalModel._eval_acc())

    evalModel = EvalModel(backdoor_model, clean_testset, device)
    print("No defence CleanACC:",evalModel._eval_acc())