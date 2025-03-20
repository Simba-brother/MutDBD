import os
import time
import torch
import torch.nn as nn
import setproctitle
from torch.utils.data import DataLoader
from codes import config
from codes.asd import defence_train
from codes.scripts.dataset_constructor import *
from codes.models import get_model
from codes.common.eval_model import EvalModel

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

# transform数据集
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets, gtsrb_IAD, gtsrb_Refool, gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets, imagenet_IAD, imagenet_Refool, imagenet_WaNet

# from codes.tools import model_train_test


# 进程名称
proctitle = f"ASD|{config.dataset_name}|{config.model_name}|{config.attack_name}"
setproctitle.setproctitle(proctitle)
print(proctitle)

# 加载后门攻击配套数据

backdoor_data = torch.load(os.path.join(config.exp_root_dir, "ATTACK", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth"), map_location="cpu")
# backdoor_model = backdoor_data["backdoor_model"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_testset = backdoor_data["poisoned_testset"] # fixed

victim_model = get_model(dataset_name=config.dataset_name, model_name=config.model_name)

# 根据poisoned_ids得到非预制菜poisoneds_trainset
if config.dataset_name == "CIFAR10":
    if config.attack_name == "BadNets":
        poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = cifar10_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, clean_testset = cifar10_WaNet()
elif config.dataset_name == "GTSRB":
    if config.attack_name == "BadNets":
        poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = gtsrb_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, clean_testset = gtsrb_WaNet()
elif config.dataset_name == "ImageNet2012_subset":
    if config.attack_name == "BadNets":
        poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_BadNets()
    elif config.attack_name == "IAD":
        poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(config.model_name,poisoned_ids,"train")
        clean_trainset, _, clean_testset, _ = imagenet_IAD()
    elif config.attack_name == "Refool":
        poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_Refool()
    elif config.attack_name == "WaNet":
        poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(config.model_name, poisoned_ids,"train")
        clean_trainset, clean_testset = imagenet_WaNet()
# 数据加载器
poisoned_trainset_loader = DataLoader(
            poisoned_trainset, # 新鲜
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

poisoned_evalset_loader = DataLoader(
            poisoned_trainset, # 新鲜
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

clean_testset_loader = DataLoader(
            clean_testset, # 新鲜
            batch_size=64, 
            shuffle=False,
            num_workers=4,
            pin_memory=True)

poisoned_testset_loader = DataLoader(
            poisoned_testset, # 预制
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
# 获得设备
device = torch.device(f"cuda:{config.gpu_id}")

# 开始防御式训练
print("开始ASD防御式训练")
time_1 = time.perf_counter()
best_ckpt_path, latest_ckpt_path = defence_train(
        model = victim_model, # victim model
        class_num = config.class_num, # 分类数量
        poisoned_train_dataset = poisoned_trainset, # 有污染的训练集
        poisoned_ids = poisoned_ids, # 被污染的样本id list
        poisoned_eval_dataset_loader = poisoned_evalset_loader, # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader = poisoned_trainset_loader, # 有污染的训练集加载器（打乱加载）
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
em = EvalModel(new_model,clean_testset,torch.device(f"cuda:{config.gpu_id}"),)
clean_test_acc = em.eval_acc()
'''
clean_test_acc = model_train_test.test(
    model = new_model,
    testset = clean_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
'''
# (2) 评估新模型在poisoned testset上的acc
em = EvalModel(new_model,poisoned_testset,torch.device(f"cuda:{config.gpu_id}"),)
poisoned_test_acc = em.eval_acc()
'''
poisoned_test_acc = model_train_test.test(
    model = new_model,
    testset = poisoned_testset,
    batch_size = 128,
    device = torch.device(f"cuda:{config.gpu_id}"),
    loss_fn = nn.CrossEntropyLoss()
    )
'''
print({'clean_test_acc':clean_test_acc, 'poisoned_test_acc':poisoned_test_acc})
time_4 = time.perf_counter()
print(f"评估防御结果结束，共耗时{time_4-time_2}秒")
