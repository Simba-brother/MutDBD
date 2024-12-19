
import os
import time

import torch

from codes import config
from codes.common.eval_model import EvalModel
from codes.scripts.dataset_constructor import ExtractDataset


def update_backdoor_data(backdoor_data_path):
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]

    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)
    backdoor_data["poisoned_trainset"] = poisoned_trainset
    backdoor_data["poisoned_testset"] = poisoned_testset

    # 保存数据
    torch.save(backdoor_data, backdoor_data_path)
    print("update_backdoor_data(),successful.")

def eval_backdoor(dataset_name,attack_name,model_name):
    
    backdoor_data_path = os.path.join(config.exp_root_dir, "ATTACK", dataset_name, model_name, attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    clean_testset = backdoor_data["clean_testset"]
    poisoned_ids = backdoor_data["poisoned_ids"]

    # eval
    start_time = time.time()
    device = torch.device(f"cuda:{config.gpu_id}")
    e =  EvalModel(backdoor_model,poisoned_trainset,device)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"poisoned_trainset_acc:{acc},cost time:{end_time-start_time:.1f}")

    start_time = time.time()
    device = torch.device(f"cuda:{config.gpu_id}")
    e =  EvalModel(backdoor_model,poisoned_testset,device)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"poisoned_testset(ASR):{acc},cost time:{end_time-start_time:.1f}")

    start_time = time.time()
    device = torch.device(f"cuda:{config.gpu_id}")
    e =  EvalModel(backdoor_model,clean_testset,device)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"clean_testset(Clean ACC):{acc},cost time:{end_time-start_time:.1f}")

