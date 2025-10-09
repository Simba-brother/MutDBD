
import os
import setproctitle
import torch
from codes import config
from codes.scripts.dataset_constructor import *
from codes.tools import EvalModel


def check_poisoned_testset(dataset):
    N = len(dataset)
    label_list = []
    for i in range(N):
        sample,label,isPoisoned =  dataset[i]
        label_list.append(label)
    label_set = set(label_list)
    if len(label_set) == 1 and list(label_set)[0] == 3:
        return "pass"
    return "No pass"
def check_poisoned_trainset(dataset,poisoned_ids):
    isPoisoned_list = []
    label_list = []
    for i in poisoned_ids:
        sample, label, isPoisoned = dataset[i]
        isPoisoned_list.append(isPoisoned)
        label_list.append(label)
    if len(isPoisoned_list) == len(poisoned_ids) and len(set(isPoisoned_list)) == 1 and list(set(isPoisoned_list))[0] == True:
        if len(set(label_list)) == 1 and list(set(label_list))[0] == 3:
            return "pass"
    else:
        return "No pass"


if __name__ == "__main__":

    proctitle = f"EvalBackdoor|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    print(f"proctitle:{proctitle}")
    # 获得backdoor_data
    backdoor_data_path = os.path.join(config.exp_root_dir, 
                                    "ATTACK", 
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

    # 检验poisoned_testset的label
    check_res_testset = check_poisoned_testset(poisoned_testset)
    if check_res_testset == "No pass":
        print("check_res_testset No pass")
    else:
        check_res_trainset =  check_poisoned_trainset(poisoned_trainset,poisoned_ids)
        if check_res_trainset == "No pass":
            print("check_res_trainset No pass")
        else:
            device = torch.device(f"cuda:{config.gpu_id}")
            evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
            print("ASR:",evalModel.eval_acc())
            evalModel = EvalModel(backdoor_model, clean_testset, device, batch_size=512, num_workers=4)
            print("CleanAcc:",evalModel.eval_acc())