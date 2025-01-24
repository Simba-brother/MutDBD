import os
import time
import torch
from torch.utils.data import DataLoader
import config
from codes.asd import defence_train
from codes.scripts.dataset_constructor import *
from codes.models import get_model

from codes.tools import model_train_test
import torch.nn as nn
import setproctitle

# 进程名称
proctitle = f"ASD|{config.dataset_name}|{config.model_name}|{config.attack_name}"
setproctitle.setproctitle(proctitle)
print(proctitle)

# 加载后门攻击配套数据

backdoor_data = torch.load(os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth"), map_location="cpu")
backdoor_model = backdoor_data["backdoor_model"]
poisoned_trainset = backdoor_data["poisoned_trainset"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_testset = backdoor_data["poisoned_testset"]
clean_testset = backdoor_data["clean_testset"]
victim_model = get_model(dataset_name=config.dataset_name, model_name=config.model_name)

# 数据加载器
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
print("开始防御式训练")
time_1 = time.perf_counter()
best_ckpt_path, latest_ckpt_path = defence_train(
        model = victim_model, # victim model
        class_num = config.class_num, # 分类数量
        poisoned_train_dataset = poisoned_trainset, # 有污染的训练集
        poisoned_ids = poisoned_ids, # 被污染的样本id list
        poisoned_eval_dataset_loader = poisoned_evalset_loader, # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader = poisoned_trainset_loader, # 有污染的训练集加载器
        clean_test_dataset_loader = clean_testset_loader, # 干净的测试集加载器
        poisoned_test_dataset_loader = poisoned_testset_loader, # 污染的测试集加载器
        device=device, # GPU设备对象
        # 实验结果存储目录
        save_dir = os.path.join(config.exp_root_dir, 
                "ASD", 
                config.dataset_name, 
                config.model_name, 
                config.attack_name, 
                time.strftime("%Y-%m-%d_%H:%M:%S")
                ),
        dataset_name = config.dataset_name,
        model_name = config.model_name,
        )
time_2 = time.perf_counter()
print(f"防御式训练完成，共耗时{time_2-time_1}秒")
# 评估防御结果
print("开始评估防御结果")
time_3 = time.perf_counter()
best_model_ckpt = torch.load(best_ckpt_path, map_location="cpu")
victim_model.load_state_dict(best_model_ckpt["model_state_dict"])
new_model = victim_model
# (1) 评估新模型在clean testset上的acc
clean_test_acc = model_train_test.test(
    model = new_model,
    testset = clean_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
# (2) 评估新模型在poisoned testset上的acc
poisoned_test_acc = model_train_test.test(
    model = new_model,
    testset = poisoned_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
print({'clean_test_acc':clean_test_acc, 'poisoned_test_acc':poisoned_test_acc})
time_4 = time.perf_counter()
print(f"评估防御结果结束，共耗时{time_4-time_2}秒")
