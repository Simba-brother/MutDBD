
'''
主要用于CIFAR10和GTSRB数据集的backdoor_data的更新和评估测试'''
import os
import time

import torch

from codes import config
from codes.common.eval_model import EvalModel
from codes.scripts.dataset_constructor import ExtractDataset


def get_classes_rank(dataset_name, model_name, attack_name, exp_root_dir)->list:
    '''获得类别排序'''
    mutated_rate = 0.01
    measure_name = "Precision_mean"
    if dataset_name in ["CIFAR10","GTSRB"]:
        grid = joblib.load(os.path.join(exp_root_dir,"grid.joblib"))
        classes_rank = grid[dataset_name][model_name][attack_name][mutated_rate][measure_name]["class_rank"]
    elif dataset_name == "ImageNet2012_subset":
        classRank_data = joblib.load(os.path.join(
            exp_root_dir,
            "ClassRank",
            dataset_name,
            model_name,
            attack_name,
            str(mutated_rate),
            measure_name,
            "ClassRank.joblib"
        ))
        classes_rank =classRank_data["class_rank"]
    else:
        raise Exception("数据集名称错误")
    return classes_rank

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

def eval_backdoor(dataset_name,attack_name,model_name, clean_testset=None):
    
    backdoor_data_path = os.path.join(config.exp_root_dir, "ATTACK", dataset_name, model_name, attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    # poisoned_trainset = backdoor_data["poisoned_trainset"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    # if clean_testset is None:
    #     clean_testset = backdoor_data["clean_testset"]
    # poisoned_ids = backdoor_data["poisoned_ids"]

    # eval
    start_time = time.time()
    device = torch.device(f"cuda:{config.gpu_id}")
    '''
    e =  EvalModel(backdoor_model,poisoned_trainset,device,batch_size=128,num_workers=4)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"poisoned_trainset_acc:{acc},cost time:{end_time-start_time:.1f}")
    '''

    start_time = time.time()
    e =  EvalModel(backdoor_model,poisoned_testset,device,batch_size=128,num_workers=4)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"poisoned_testset(ASR):{acc},cost time:{end_time-start_time:.1f}")

    start_time = time.time()
    e =  EvalModel(backdoor_model,clean_testset,device,batch_size=128,num_workers=4)
    acc = e.eval_acc()
    end_time = time.time()
    print(f"clean_testset(Clean ACC):{acc},cost time:{end_time-start_time:.1f}")

