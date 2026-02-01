import os
import math
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split
from mid_data_loader import get_backdoor_data, get_class_rank
from models.model_loader import get_model
from datasets.posisoned_dataset import get_all_dataset
import torch.nn as nn
import torch
from utils.common_utils import Record,convert_to_hms,get_formattedDateTime
import queue
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import random


def clean_seed(poisoned_trainset,poisoned_ids,strict_clean:bool=True):
    '''
    选择干净种子
    '''
    # 数据加载器
    poisoned_evalset_loader = DataLoader(
                poisoned_trainset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    # 获得种子
    # {class_id:[sample_id]}
    clean_sample_dict = defaultdict(list)
    label_list = []
    for _, batch in enumerate(poisoned_evalset_loader):
        Y = batch[1]
        label_list.extend(Y.tolist())

    for sample_id in range(len(poisoned_trainset)):
        if sample_id not in poisoned_ids:
            label = label_list[sample_id]
            clean_sample_dict[label].append(sample_id)

    # 获得种子数据集
    seed_sample_id_list = []
    for class_id,sample_id_list in clean_sample_dict.items():
        seed_sample_id_list.extend(np.random.choice(sample_id_list, replace=False, size=10).tolist())
    if strict_clean is False:
        # 种子种混入了2个poisoned samples
        # 随机删除2个clean seed id
        first_element = random.choice(seed_sample_id_list)
        seed_sample_id_list.remove(first_element)
        second_element = random.choice(seed_sample_id_list)
        seed_sample_id_list.remove(second_element)
        # 从p ids 中随机找2个id 放入 seed_sample_id_list
        p_ids = random.sample(poisoned_ids, 2)
        seed_sample_id_list.extend(p_ids)
    seed_p_id_set = set(seed_sample_id_list) & set(poisoned_ids)
    clean_seedSet = Subset(poisoned_trainset,seed_sample_id_list)
    return clean_seedSet,seed_sample_id_list

def sigmoid(x: float) -> float:
    # stable for large |x|
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def resort(ranked_sample_id_list,label_list,class_rank:list,beta:float=1.0,sigmoid_flag:bool=False)->list:
        # 基于class_rank得到每个类别权重，原则是越可疑的类别（索引越小的类别），权（分）越大
        cls_num = len(class_rank)
        cls2score = {}
        for idx, cls in enumerate(class_rank):
            cls2score[cls] = (cls_num - beta*idx)/cls_num  # 类别3：(10-0)/10 = 1, (10-9)/ 10 = 0.1
        sample_num = len(ranked_sample_id_list)
        # 一个优先级队列
        q = queue.PriorityQueue()
        for idx, sample_id in enumerate(ranked_sample_id_list):
            sample_rank = idx+1
            sample_label = label_list[sample_id]
            cls_score = cls2score[sample_label]
            if sigmoid_flag is True:
                cls_score = sigmoid(cls_score)
            score = (sample_rank/sample_num)*cls_score # cls_score 归一化了，没加log
            q.put((score,sample_id)) # 越小优先级越高，越干净
        resort_sample_id_list = []
        while not q.empty():
            resort_sample_id_list.append(q.get()[1])
        return resort_sample_id_list

def sort_sample_id(model,
                   device,
                   poisoned_trainset,
                   poisoned_ids,
                   class_rank=None,
                   beta:float = 1.0,
                   sigmoid_flag:bool = False):
    '''基于模型损失值或class_rank对样本进行可疑程度排序'''
    model.to(device)
    dataset_loader = DataLoader(poisoned_trainset,batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    # 损失函数
    # loss_fn = SCELoss(num_classes=10, reduction="none") # nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    loss_record = Record("loss", len(dataset_loader.dataset)) # 记录每个样本的loss
    label_record = Record("label", len(dataset_loader.dataset))
    model.eval()
    # 判断模型是在CPU还是GPU上
    for _, batch in enumerate(dataset_loader): # 分批次遍历数据加载器
        # 该批次数据
        X = batch[0].to(device)
        # 该批次标签
        Y = batch[1].to(device)
        with torch.no_grad():
            P_Y = model(X)
        loss_fn.reduction = "none" # 数据不进行规约,以此来得到每个样本的loss,而不是批次的avg_loss
        loss = loss_fn(P_Y, Y)
        loss_record.update(loss.cpu())
        label_record.update(Y.cpu())
    # 基于loss排名
    loss_array = loss_record.data.numpy()
    # 基于loss的从小到大的样本本id排序数组
    based_loss_ranked_sample_id_list =  loss_array.argsort().tolist()
    
    if class_rank is None:
        ranked_sample_id_list = based_loss_ranked_sample_id_list
    else:
        label_list = label_record.data.numpy().tolist()
        ranked_sample_id_list  = resort(based_loss_ranked_sample_id_list,label_list,
                                        class_rank,beta,sigmoid_flag)
    # 获得对应的poisoned_flag
    isPoisoned_list = []
    for sample_id in ranked_sample_id_list:
        if sample_id in poisoned_ids:
            isPoisoned_list.append(True)
        else:
            isPoisoned_list.append(False)
    return ranked_sample_id_list, isPoisoned_list,loss_array

def chose_retrain_set(ranker_model, device, 
                      choice_rate, poisoned_trainset, poisoned_ids,
                      class_rank=None,beta:float = 1.0,sigmoid_fag:bool=False):
    '''
    选择用于后门模型重训练的数据集
    '''
    ranked_sample_id_list, isPoisoned_list,loss_array = sort_sample_id(
                                                ranker_model,
                                                device,
                                                poisoned_trainset,
                                                poisoned_ids,
                                                class_rank,
                                                beta,
                                                sigmoid_fag)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    choiced_num = len(choiced_sample_id_list)
    PN = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            PN += 1
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)
    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN

def draw(isPoisoned_list_1, isPoisoned_list_2 ,file_name):
    '''
    论文配图，动机章节：热力图
    # 话图看一下中毒样本在序中的分布
    distribution = [1 if flag else 0 for flag in isPoisoned_list]
    # 绘制热力图
    # 创建图形时设置较小的高度
    plt.style.use(['science','ieee'])
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(3, 0.5))  # 宽度为10，高度为2（可根据需要调整）
    plt.imshow([distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    # plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('ranking',fontsize='3')
    # 调整横轴刻度字号
    plt.xticks(fontsize=3)  # 明确设置横轴刻度字号为6pt
    # plt.colorbar()
    plt.yticks([])
    plt.savefig(f"imgs/sample_sort/{file_name}", bbox_inches='tight', dpi=800) # pad_inches=0.0
    plt.close()
    '''
    plt.style.use('science')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6
    })
    distribution1 = [1 if flag else 0 for flag in isPoisoned_list_1]
    distribution2 = [1 if flag else 0 for flag in isPoisoned_list_2]
    
    # 创建2行1列的子图
    fig, axs = plt.subplots(2, 1, figsize=(3, 1.0))  # 总高度调整为1.0，每个子图高度约0.5

    # 确保axs是数组形式（即使只有一行）
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # 绘制第一个子图
    axs[0].imshow([distribution1], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[0].set_xlabel('Sample ranking', fontsize=3)
    axs[0].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[0].set_yticks([])

    # 绘制第二个子图
    axs[1].imshow([distribution2], aspect='auto', cmap='Reds', interpolation='nearest')
    axs[1].set_xlabel('Sample ranking', fontsize=3)
    axs[1].tick_params(axis='x', labelsize=3)  # 修正：使用tick_params设置刻度标签字号
    axs[1].set_yticks([])

    # 调整子图间距
    plt.subplots_adjust(hspace=0.3)  # 调整垂直间距

    # 保存为高分辨率图像
    plt.savefig(f"imgs/Motivation/SampleRanking/{file_name}", 
                bbox_inches='tight', 
                pad_inches=0.02,
                dpi=800,
                facecolor='white',
                edgecolor='none')

    plt.close()

def main_one_sence(dataset_name,model_name,attack_name,beta:float, sigmoid_flag:bool):
    blank_model = get_model(dataset_name, model_name)
    ranker_model_state_dict = torch.load(ranker_model_state_dict_path,map_location="cpu")
    blank_model.load_state_dict(ranker_model_state_dict)
    ranker_model = blank_model
    choice_rate = 0.6
    
    class_rank = get_class_rank(dataset_name,model_name,attack_name)
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = \
        chose_retrain_set(ranker_model,device,choice_rate,poisoned_trainset,poisoned_ids,
                          class_rank,beta,sigmoid_flag)

    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN
    
def atomic_json_dump(obj, out_path, indent=2):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(out_path)

def load_results(exp_save_path):
    if not os.path.exists(exp_save_path):
        return {}
    with open(exp_save_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_experiment_result(exp_save_path, dataset_name, model_name, attack_name,
                          r_seed, beta, result_data):
    """
    保存单个实验结果到嵌套JSON
    结构: {dataset: {model: {attack: {beta: {r_seed: result}}}}}
    """
    # 加载现有数据
    data = load_results(exp_save_path)

    # 构建嵌套结构
    if dataset_name not in data:
        data[dataset_name] = {}
    if model_name not in data[dataset_name]:
        data[dataset_name][model_name] = {}
    if attack_name not in data[dataset_name][model_name]:
        data[dataset_name][model_name][attack_name] = {}
    if str(beta) not in data[dataset_name][model_name][attack_name]:
        data[dataset_name][model_name][attack_name][str(beta)] = {}

    # 保存结果
    data[dataset_name][model_name][attack_name][str(beta)][str(r_seed)] = result_data

    # 原子写入
    atomic_json_dump(data, exp_save_path)


def get_experiment_result(dataset_name, model_name, attack_name, r_seed, beta):
    """
    获取特定场景的实验结果
    """
    data = load_results()

    try:
        return data[dataset_name][model_name][attack_name][str(beta)][str(r_seed)]
    except KeyError:
        return None

def get_backdoor_model(dataset_name, model_name, backdoor_data):
    
    # 后门模型
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    return backdoor_model


if __name__ == "__main__":
    '''
    # 单场景
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "IAD"
    device = torch.device("cuda:0")
    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
    # 后门模型
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)


    ranker_model_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours",dataset_name,model_name,attack_name,
                                                "exp_1","best_BD_model.pth")
    main_one_sence()
    '''


    # 全场景
    cur_pid = os.getpid()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10","GTSRB","ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    beta_list = [1.0, 0.75, 0.5, 0.25, 0.0]
    r_seed_list = list(range(1,11)) # 1-10
    sigmoid_flag = False
    device = torch.device("cuda:0")
    exp_save_dir = os.path.join(exp_root_dir, "Exp_Results", "discussion_beta")
    os.makedirs(exp_save_dir,exist_ok=True)
    exp_result_file_name = f"results.json"
    exp_save_path = os.path.join(exp_save_dir,exp_result_file_name)

    print("cur_pid:",cur_pid)
    print("exp_root_dir:",exp_root_dir)
    print("dataset_name_list:",dataset_name_list)
    print("model_name_list:",model_name_list)
    print("attack_name_list:",attack_name_list)
    print("beta_list:",beta_list)
    print("r_seed_list:",r_seed_list)
    print("sigmoid_flag:",sigmoid_flag)
    print("exp_save_path:",exp_save_path)

    total_start_time = time.perf_counter()
    for r_seed in r_seed_list:
        for dataset_name in dataset_name_list:
            for model_name in model_name_list:
                for attack_name in attack_name_list:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
                    backdoor_model = get_backdoor_model(dataset_name, model_name, backdoor_data)
                    ranker_model_state_dict_path = os.path.join(
                            exp_root_dir,"Defense","Ours",dataset_name,model_name,attack_name,
                            f"exp_{r_seed}","best_BD_model.pth")
                    backdoor_model = backdoor_model.to(device).eval()
                    # 训练数据集中中毒样本id
                    poisoned_ids = backdoor_data["poisoned_ids"]
                    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
                    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
                    for beta in beta_list:
                        start_time = time.perf_counter()
                        print(f"\n{dataset_name}|{model_name}|{attack_name}|beta={beta}|r_seed={r_seed}")
                        choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = \
                            main_one_sence(dataset_name,model_name,attack_name,beta,sigmoid_flag)
                        choiced_ids = [int(i) for i in list(choiced_sample_id_list)]
                        print(f"\tPN:{int(PN)}")
                        res_data = {
                            "PN": int(PN),
                        }
                        # 立即保存到嵌套JSON
                        save_experiment_result(
                            exp_save_path, dataset_name, model_name, attack_name,
                            r_seed, beta, res_data
                        )
                        end_time = time.perf_counter()
                        cost_time = end_time - start_time
                        hours, minutes, seconds = convert_to_hms(cost_time)
                        print(f"\t耗时:{hours}时{minutes}分{seconds:.1f}秒")

    total_end_time = time.perf_counter()
    total_cost_time = total_end_time - total_start_time
    hours, minutes, seconds = convert_to_hms(total_cost_time)
    print(f"\n全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
    

  