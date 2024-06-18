import os
import setproctitle
import torch
# from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop
import torch.nn as nn
from codes import models
from codes import config
from codes.ourMethod import TargetClassProcessor,detect_poisonedAndclean_from_targetClass
from codes.ourMethod import defence_train
from codes.scripts.dataset_constructor import *
from codes.tools import model_train_test
from codes import utils
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
victim_model = models.get_resnet(18,10)
# victim_model = models.get_resnet_cifar(10)

# 第一步:确定target class
target_class_processor = TargetClassProcessor(
    dataset_name = config.dataset_name, # 数据集名称
    model_name = config.model_name, # 模型名称
    attack_name = config.attack_name, # 攻击名称
    mutation_name_list = config.mutation_name_list, # 变异算子名称list
    mutation_rate_list = config.mutation_rate_list, # 变异率list
    exp_root_dir = config.exp_root_dir, # 实验数据根目录
    class_num = config.class_num, # 数据集的分类数
    mutated_model_num = config.mutation_model_num, # 每个变异率下变异模型的数量,eg:50
    mutation_operator_num = len(config.mutation_name_list)  # 变异算子的数量 = len(mutation_name_list),eg:5
    )
dic_1, dic_2 = target_class_processor.get_adaptive_rate_of_Hybrid_mutator()
adaptive_mutation_rate = dic_1["adaptive_rate"]
target_class_i = dic_1["target_class_i"]
# 第二步:从target class中检测木马样本
priority_list, target_class_clean_set, purePoisonedTrainDataset = detect_poisonedAndclean_from_targetClass(adaptive_mutation_rate)
no_targetClass_dataset = ExtractNoTargetClassDataset(poisoned_trainset, target_class_i)
# 第三步:获得清洗后的训练集
new_train_dataset = defence_train.get_train_dataset(
    priority_list = priority_list, 
    cut_off = 0.5, 
    target_class_clean_set = target_class_clean_set, 
    purePoisonedTrainDataset = purePoisonedTrainDataset, 
    no_target_class_dataset =no_targetClass_dataset
    )
# 第四步:开始训练
train_ans = model_train_test.train(
    model = victim_model,
    trainset = new_train_dataset,
    epochs = 200,
    batch_size = 128,
    optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
    loss_fn = nn.CrossEntropyLoss(),
    device = torch.device(f"cuda:{config.gpu_id}"),
    work_dir = os.path.join("OurMethod", f"{config.dataset_name}", f"{config.model_name}", f"{config.attack_name}", "defence"),
    scheduler = None
)
# 第五步：评估新模型
new_model = train_ans["best_model"]
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
