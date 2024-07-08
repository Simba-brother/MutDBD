# 加载dict_state
import joblib
import os
import torch.nn as nn
from codes import config
from codes.scripts.dataset_constructor import *
from codes.tools.model_train_test import test

dict_state_path = os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "attack", "dict_state.pth")

dict_state =  torch.load(dict_state_path, map_location="cpu")
backdoor_model = dict_state["backdoor_model"]
poisoned_trainset = dict_state["poisoned_trainset"]
poisoned_ids = dict_state["poisoned_ids"]
poisoned_testset = dict_state["poisoned_testset"]
clean_testset = dict_state["clean_testset"]

backdoor_clean_acc = test(
    model = backdoor_model,
    testset = clean_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
backdoor_poisoned_acc = test(
    model = backdoor_model,
    testset = poisoned_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )

# ASD防御模型评估
asd_defence_model_weights_path = os.path.join(config.exp_root_dir, "ASD", config.dataset_name, config.model_name, config.attack_name, "ckpt", "best_model.pt")
defence_model =  dict_state["backdoor_model"]
defence_model.load_state_dict(torch.load(asd_defence_model_weights_path)["model_state_dict"])
asd_defence_clean_acc = test(
    model = defence_model,
    testset = clean_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
asd_defence_poisoned_acc = test(
    model = defence_model,
    testset = poisoned_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
print(f"backdoor_model: clean_acc:{backdoor_clean_acc}, poisoned_acc:{backdoor_poisoned_acc}")
print(f"defence_model: clean_acc:{asd_defence_clean_acc}, poisoned_acc:{asd_defence_poisoned_acc}")

