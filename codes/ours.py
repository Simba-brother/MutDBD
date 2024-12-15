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

from codes.ourMethod import (
        # 模型变异class
        ModelMutat,
        # 检测target class方法,并获得adaptive mutation rate
        detect_target_class,
        # 对变异模型进行排序
        sort_mutated_models,
        # 从target class中检测clean与poisoned
        detect_poisonedAndclean_from_targetClass,
        # 构建新数据集方法
        newdataset_construct,
    )
from codes.utils import create_dir
import joblib

def gen_mutaion_models():
    '''生成变异模型
    Return:
        mutated_weights_dir
    '''
    mutated_weights_dir = os.path.join(config.exp_root_dir,config.dataset_name,config.model_name,config.attack_name,"mutation_models", time.strftime("%Y-%m-%d_%H:%M:%S"))
    # 遍历变异率
    for mutation_rate in config.mutation_rate_list:
        print(f"current_mutation_rate:{mutation_rate}")
        # 获得模型变异对象
        mm = ModelMutat(original_model=backdoor_model, mutation_rate=mutation_rate)
        # 遍历变异算子：每个变异率下会有5个变异算子
        for mutation_name in config.mutation_name_list:
            # 遍历变异次数：每个变异算子会生成50个变异体
            for i in range(config.mutation_model_num):
                if mutation_name == "gf":
                    mutated_model = mm._gf_mut(scale=5)
                if mutation_name == "neuron_activation_inverse":
                    mutated_model = mm._neuron_activation_inverse()
                if  mutation_name == "neuron_block":
                    mutated_model = mm._neuron_block()
                if mutation_name == "neuron_switch":
                    mutated_model = mm._neuron_switch()
                if mutation_name == "weight_shuffle":
                    mutated_model = mm._weight_shuffling()
                save_dir = os.path.join(mutated_weights_dir,str(mutation_rate),mutation_name)
                create_dir(save_dir)
                save_file_name = f"mutated_weights_{i}.pth"
                save_file_path = os.path.join(save_dir,save_file_name)
                torch.save(mutated_model.state_dict(), save_file_path)
    return mutated_weights_dir

def eval_mutation_models(mutated_weights_dir,device):
    '''
    评估变异模型
    Args:
        mutated_weights_dir
    Return:
        dict_eval_report:{
                mutate_rate:{
                    "gf":[report_1, ...,]
            }
    '''
    dict_eval_report = defaultdict(dict)
    for mutate_rate in config.mutation_rate_list:
        dict_eval_report[mutate_rate] = defaultdict(list)
        for mutator_name in config.mutation_name_list:
            for i in range(config.mutation_model_num):
                weights_path = os.path.join(mutated_weights_dir,str(mutate_rate), mutator_name, f"mutated_weights_{i}.pth")
                weights = torch.load(weights_path,map_location="cpu")
                victim_model.load_state_dict(weights)
                evalModel = EvalModel(victim_model, poisoned_trainset, device=device)
                report = evalModel._eval_classes_acc()
                dict_eval_report[mutate_rate][mutator_name].append(report)
    return dict_eval_report

if __name__ == "__main__":
    # 进程名称
    proctitle = f"Ours|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    print(f"proctitle:{proctitle}")
    # 获得backdoor_data
    backdoor_data_path = os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    poisoned_testset =backdoor_data["poisoned_testset"]
    poisoned_ids =backdoor_data["poisoned_ids"]
    clean_testset =backdoor_data["clean_testset"]
    # 数据预transform,为了后面训练加载的更快
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset,poisoned_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset,poisoned_ids)
    poisoned_testset = ExtractDataset(poisoned_testset)
    # victim model
    victim_model = backdoor_model
    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device)
    print("No defence ASR:",evalModel._eval_acc())
    # 生成变异模型
    print("开始生成变异模型")
    start_1_time = time.perf_counter()
    mutated_weights_dir = gen_mutaion_models()
    end_1_time = time.perf_counter()
    cost_1_time = end_1_time - start_1_time
    print("生成的变异模型权重保存位置:", mutated_weights_dir)
    print(f"生成变异模型结束,共耗时{cost_1_time}s")

    # 评估变异模型
    print("开始在poisoned_trainset上评估变异模型")
    start_2_time = time.perf_counter()
    dict_eval_report = eval_mutation_models(mutated_weights_dir,device=torch.device(f"cuda:{config.gpu_id}"))
    save_dir = os.path.join(config.exp_root_dir, config.dataset_name, config.model_name, config.attack_name)
    create_dir(save_dir)
    save_file_name = "dict_eval_report.data"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(dict_eval_report, save_path)
    end_2_time = time.perf_counter()
    cost_2_time = end_2_time - start_2_time
    print("评估的变异模型结果保存位置:", save_path)
    print(f"评估变异模型结束,共耗时{cost_2_time}s")

    # 第一步:确定target class和adaptive mutation rate
    print("第一步开始:确定target class和adaptive mutation rate")
    dict_eval_report_path = os.path.join(
        config.exp_root_dir, config.dataset_name, config.model_name, config.attack_name, "dict_eval_report.data"
    )
    dict_eval_report = joblib.load(dict_eval_report_path)
    start_3_time = time.perf_counter() 
    target_class_i, adaptive_rate = detect_target_class(dict_eval_report,config.class_num)
    print(f"adaptive_rate:{adaptive_rate},target_class_i:{target_class_i}")
    end_3_time = time.perf_counter()
    cost_3_time = end_3_time - start_3_time
    # wandb.log({'adaptive_rate':adaptive_rate, 'target_class_i':target_class_i})
    print(f"第一步结束。共耗时:{cost_3_time}s")
    if target_class_i != 1:
        print(f"确定target class错误，target class:{target_class_i}")
        sys.exit()

    # mutated_weights_dir = "/data/mml/backdoor_detect/experiments/CIFAR10/DenseNet/IAD/mutation_models/2024-07-16_13:17:14"
    # adaptive_rate = 0.01
    # target_class_i = 1

    # 第二步:从target class中检测木马样本
    print("第二步开始:从target class中检测木马样本")

    start_4_time = time.perf_counter()
    # 排序变异模型
    # 获得apative_rate下面所有变异算子的权重路径list
    mutation_weights_path_list = []
    mutation_models_dir = os.path.join(mutated_weights_dir,str(adaptive_rate))
    for mutation_name in config.mutation_name_list:
        for i in range(config.mutation_model_num):
            mutation_weights_path_list.append(os.path.join(mutation_models_dir, mutation_name, f"mutated_weights_{i}.pth"))

    # target class中的clean set
    target_class_clean_set = ExtractTargetClassDataset(pureCleanTrainDataset, target_class_idx = target_class_i)
    # target class中的poisoned set
    target_class_poisoned_set = ExtractTargetClassDataset(purePoisonedTrainDataset, target_class_idx = target_class_i)
    sorted_weights_path_list = sort_mutated_models(
            model_struct = victim_model,
            mutation_weights_path_list = mutation_weights_path_list,
            target_class_clean_set = target_class_clean_set,
            target_class_poisoned_set = target_class_poisoned_set,
            device = torch.device(f"cuda:{config.gpu_id}")
    )
    priority_list, target_class_clean_set, purePoisonedTrainDataset = detect_poisonedAndclean_from_targetClass(
            sorted_weights_path_list,
            model_struct = victim_model,
            target_class_clean_set = target_class_clean_set,
            purePoisonedTrainDataset = target_class_poisoned_set
    )
    end_4_time = time.perf_counter()
    cost_4_time = end_4_time - start_4_time
    print(f"第二步结束。共耗时:{cost_4_time}s")

    # 第三步:获得清洗后的训练集
    print("第三步开始:获得清洗后的训练集")
    start_5_time = time.perf_counter()
    no_targetClass_dataset = ExtractNoTargetClassDataset(poisoned_trainset, target_class_i)
    new_train_dataset = newdataset_construct.get_train_dataset(
        priority_list = priority_list, 
        cut_off = 0.5, # importent!!!
        target_class_clean_set = target_class_clean_set, 
        purePoisonedTrainDataset = purePoisonedTrainDataset, 
        no_target_class_dataset =no_targetClass_dataset
        )
    end_5_time = time.perf_counter()
    cost_5_time = end_5_time - start_5_time
    print(f"第三步结束。共耗时{cost_5_time}s")

    # 第四步:开始训练
    print("第四步开始:在清洗后的训练集上训练")
    start_6_time = time.perf_counter()
    victim_model = models.get_model(dataset_name=config.dataset_name,model_name=config.model_name)
    train_ans = model_train_test.train(
        model = victim_model,
        trainset = new_train_dataset,
        epochs = 120,
        batch_size = 128,
        optimizer = torch.optim.SGD(victim_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
        init_lr = 0.1,
        loss_fn = nn.CrossEntropyLoss(),
        device = torch.device(f"cuda:{config.gpu_id}"),
        work_dir = os.path.join(config.exp_root_dir, config.dataset_name, config.model_name, config.attack_name, "defence", "OurMethod"),
        scheduler = None
    )
    end_6_time = time.perf_counter()
    cost_6_time = end_6_time - start_6_time
    print(f"第四步结束。共耗时:{cost_6_time}s")

    # 第五步：评估新模型
    print("第五步开始：评估新模型")
    start_7_time = time.perf_counter()
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
    end_7_time = time.perf_counter()
    cost_7_time  = end_7_time - start_7_time
    print(f"第五步结束。共耗时:{cost_7_time}s")
    print({'clean_test_acc':clean_test_acc, 'poisoned_test_acc':poisoned_test_acc})

