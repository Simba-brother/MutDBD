import os
import torch
from torch.utils.data import DataLoader
import config
from asd import defence_train
from scripts.dataset_constructor import *
import models
import setproctitle

# 进程名称
setproctitle.setproctitle(f"ASD|{config.dataset_name}|{config.model_name}|{config.attack_name}")
# 获得数据和模型
dict_state_path = os.path.join(config.exp_root_dir,"attack",config.dataset_name,config.model_name,config.attack_name, "attack","dict_state.pth")
dict_state = torch.load(dict_state_path, map_location="cpu")
backdoor_model = dict_state["backdoor_model"]
poisoned_trainset = dict_state["poisoned_trainset"]
poisoned_ids = dict_state["poisoned_ids"]
poisoned_testset = dict_state["poisoned_testset"]
clean_testset = dict_state["clean_testset"]
# victim_model = models.get_resnet(18,10)
victim_model = models.get_resnet_cifar(10)

poisoned_trainset_loader = DataLoader(
            poisoned_trainset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
poisoned_evalset_loader = DataLoader(
            poisoned_trainset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
clean_testset_loader = DataLoader(
            clean_testset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
poisoned_testset_loader = DataLoader(
            poisoned_testset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
# 获得设备
device = torch.device(f"cuda:{config.gpu_id}")
# 开始防御式训练
defence_train(
        model = victim_model, # victim model
        class_num = config.class_num, # 分类数量
        poisoned_train_dataset = poisoned_trainset, # 有污染的训练集
        poisoned_ids = poisoned_ids, # 被污染的样本id list
        poisoned_eval_dataset_loader = poisoned_evalset_loader, # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader = poisoned_trainset_loader, #有污染的训练集加载器
        clean_test_dataset_loader = clean_testset_loader, # 干净的测试集加载器
        poisoned_test_dataset_loader = poisoned_testset_loader, # 污染的测试集加载器
        device=device, # GPU设备对象
        save_dir = os.path.join(config.exp_root_dir, "ASD", config.dataset_name, config.model_name, config.attack_name)# 实验结果存储目录 save_dir = os.path.join(exp_root_dir, "ASD", dataset_name, model_name, attack_name)
        )

