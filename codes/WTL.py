'''
统计所有场景的W/T/L
'''
import os
import yaml
import torch
import pandas as pd
from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta
from codes.config import exp_root_dir,target_class_idx
from codes.models import get_model
from codes.bigUtils.dataset import get_spec_dataset
from codes.common.eval_model import EvalModel
from torch.utils.data import DataLoader,Subset
import joblib
import torch.nn as nn
import queue
from codes.asd.log import Record



def filter_poisonedSet(clean_set,poisoned_set,target_class_idx):
    # 从poisoned_testset中剔除原来就是target class的数据
    clean_testset_loader = DataLoader(
                clean_set, # 非预制
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    clean_testset_label_list = []
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(clean_set)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != target_class_idx:
            filtered_ids.append(sample_id)
    filtered_poisoned_set = Subset(poisoned_set,filtered_ids)
    return filtered_poisoned_set



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


def resort(ranked_sample_id_list,label_list,class_rank:list)->list:
        # 基于class_rank得到每个类别权重，原则是越可疑的类别（索引越小的类别），权（分）越大
        cls_num = len(class_rank)
        cls2score = {}
        for idx, cls in enumerate(class_rank):
            cls2score[cls] = (cls_num - idx)/cls_num  # 类别3：(10-0)/10 = 1, (10-9)/ 10 = 0.1
        sample_num = len(ranked_sample_id_list)
        # 一个优先级队列
        q = queue.PriorityQueue()
        for idx, sample_id in enumerate(ranked_sample_id_list):
            sample_rank = idx+1
            sample_label = label_list[sample_id]
            cls_score = cls2score[sample_label]
            score = (sample_rank/sample_num)*cls_score # cls_score 归一化了，没加log
            q.put((score,sample_id)) # 越小优先级越高，越干净
        resort_sample_id_list = []
        while not q.empty():
            resort_sample_id_list.append(q.get()[1])
        return resort_sample_id_list


def sort_sample_id(model, device, dataset_loader, class_rank=None):
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    # 记录每个样本的loss
    loss_record = Record("loss", len(dataset_loader.dataset))
    # 记录每个样本的label
    label_record = Record("label", len(dataset_loader.dataset))
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
        ranked_sample_id_list = based_loss_ranked_sample_id_list # 朴素基于损失值从小到大排序样本id
    else:
        label_list = label_record.data.numpy().tolist()
        ranked_sample_id_list  = resort(based_loss_ranked_sample_id_list,label_list,class_rank)
    '''
    # 获得对应的poisoned_flag
    isPoisoned_list = []
    for sample_id in ranked_sample_id_list:
        if sample_id in poisoned_ids:
            isPoisoned_list.append(True)
        else:
            isPoisoned_list.append(False)
    '''
    return ranked_sample_id_list

def split_method(
        ranker_model,
        poisoned_trainset,
        poisoned_ids,
        device,
        class_rank = None,
        choice_rate = 0.5
        ):
    poisoned_trainset_loader = DataLoader(
        poisoned_trainset,
        batch_size = 256,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    ranked_sample_id_list = sort_sample_id(
                            ranker_model,
                            device,
                            poisoned_trainset_loader,
                            class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    # 干净池总数
    choiced_num = len(choiced_sample_id_list)
    # 干净池中的中毒数量
    p_count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            p_count += 1
    # 干净池的中毒比例
    poisoning_rate = round(p_count/choiced_num, 2)
    return p_count, choiced_num, poisoning_rate

    '''
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)
    '''
    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list


def our_method_state(dataset_name, model_name, attack_name, random_seed):
    defensed_state_dict_path = os.path.join(exp_root_dir,"OurMethod_new", dataset_name, model_name, attack_name, f"exp_{random_seed}", "last_defense_model.pth") 
    selected_state_dict_path = os.path.join(exp_root_dir,"OurMethod_new", dataset_name, model_name, attack_name, f"exp_{random_seed}", "best_BD_model.pth") 
    return defensed_state_dict_path, selected_state_dict_path

def asd_method_state(dataset_name, model_name, attack_name, random_seed):
    # 读取yaml
    yaml_path = "codes/ASD_res_config.yaml"
    with open(yaml_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    exp_id = f"exp_{random_seed}"
    time_str = data[dataset_name][model_name][attack_name][exp_id]
    # 获得防御模型路径
    defensed_state_dict_path = os.path.join(exp_root_dir,"ASD_new",dataset_name,model_name,attack_name,time_str,"ckpt","latest_model.pt")
    # 获得选择模型路径
    selected_state_dict_path = os.path.join(exp_root_dir,"ASD_new",dataset_name,model_name,attack_name,time_str,"ckpt","secondtolast.pth")
    return defensed_state_dict_path, selected_state_dict_path



def get_res(
        defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = None, choice_rate=0.5):
    # ACC和ASR
    em = EvalModel(defence_model, clean_testset, device)
    acc = em.eval_acc()
    em = EvalModel(defence_model, filtered_poisoned_testset, device)
    asr = em.eval_acc()

    # 中毒样本切分结果
    p_num, choiced_num, poisoning_rate = split_method(
        select_model,
        poisoned_trainset,
        poisoned_ids,
        device,
        class_rank = class_rank,
        choice_rate = choice_rate
        )
    res = {"acc":acc,"asr":asr,"p_num":p_num}
    return res


def our_unit_res(dataset_name, model_name, attack_name, random_seed, 
             poisoned_trainset, poisoned_ids,
             filtered_poisoned_testset, clean_testset,
             device):
    # 过滤掉原来target class的样本
    
    # OurRes
    defensed_state_dict_path, selected_state_dict_path = our_method_state(dataset_name, model_name, attack_name, random_seed)
    defence_model = get_model(dataset_name,model_name)
    select_model = get_model(dataset_name,model_name)
    defence_model.load_state_dict(torch.load(defensed_state_dict_path,map_location="cpu"))
    select_model.load_state_dict(torch.load(selected_state_dict_path,map_location="cpu"))
    # seed微调后排序一下样本
    class_rank = get_classes_rank(dataset_name, model_name, attack_name, exp_root_dir)
    our_res = get_res(defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = class_rank, choice_rate=0.6)
    return our_res

def asd_unit_res(dataset_name, model_name, attack_name, random_seed, 
             poisoned_trainset, poisoned_ids,
             filtered_poisoned_testset, clean_testset,
             device):
    # ASDRes
    defensed_state_dict_path, selected_state_dict_path = asd_method_state(dataset_name, model_name, attack_name, random_seed)
    defence_model = get_model(dataset_name,model_name)
    select_model = get_model(dataset_name,model_name)
    defence_model.load_state_dict(torch.load(defensed_state_dict_path,map_location="cpu"))
    select_model.load_state_dict(torch.load(selected_state_dict_path,map_location="cpu"))

    ASD_res = get_res(defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = None, choice_rate=0.5)
    return ASD_res



def compare_avg(our_list, baseline_list):
    our_avg = round(sum(our_list)/len(our_list),3)
    baseline_avg = round(sum(baseline_list)/len(baseline_list),3)
    '''
    if expect == "small":
        if our_avg < baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    else:
        if our_avg > baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    '''
    return our_avg, baseline_avg





def compare_WTL(our_list, baseline_list,expect:str):
    res = ""
    # 计算W/T/L
    # Wilcoxon:https://blog.csdn.net/TUTO_TUTO/article/details/138289291
    # Wilcoxon：主要来判断两组数据是否有显著性差异。
    statistic, p_value = wilcoxon(our_list, baseline_list) # statistic:检验统计量
    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(our_list, baseline_list)
    if p_value >= 0.05:
        # 值分布没差别
        res = "Tie"
    else:
        # 值分布有差别
        if expect == "small":
            # 指标越小越好，d越接近-1越好
            if d < 0 and res != "negligible":
                res = "Win"
            elif d > 0 and res != "negligible":
                res = "Lose"
            else:
                res = "Tie"
        else:
            # 指标越大越好，d越接近1越好
            if d > 0 and res != "negligible":
                res = "Win"
            elif d < 0 and res != "negligible":
                res = "Lose"
            else:
                res = "Tie"
    return res






def main_scene():
    # 后门信息
    backdoor_data = torch.load(os.path.join(exp_root_dir, "ATTACK",
                            dataset_name, model_name, attack_name,
                            "backdoor_data.pth"), map_location="cpu")
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_testset = backdoor_data["poisoned_testset"]
    # 数据集
    poisoned_trainset, clean_trainset, clean_testset = get_spec_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    filtered_poisoned_testset = filter_poisonedSet(clean_testset,poisoned_testset,target_class_idx)
    # 10次重复实验记录
    our_acc_list = []
    our_asr_list = []
    our_p_num_list = []

    asd_acc_list = []
    asd_asr_list = []
    asd_p_num_list = []
    for random_seed in range(1,11): # 1-10
        # acc,asr,p_num
        our_res, = our_unit_res(dataset_name, model_name, attack_name, random_seed, 
                poisoned_trainset, poisoned_ids,
                filtered_poisoned_testset, clean_testset,
                device)
        asd_res, = asd_unit_res(dataset_name, model_name, attack_name, random_seed, 
                poisoned_trainset, poisoned_ids,
                filtered_poisoned_testset, clean_testset,
                device)
        our_acc_list.append(our_res["acc"])
        our_asr_list.append(our_res["asr"])
        our_p_num_list.append(our_res["p_num"])

        asd_acc_list.append(asd_res["acc"])
        asd_asr_list.append(asd_res["asr"])
        asd_p_num_list.append(asd_res["p_num"])


    our_acc_avg, asd_acc_avg = compare_avg(our_acc_list, asd_acc_list)
    our_asr_avg, asd_asr_avg = compare_avg(our_asr_list, asd_asr_list)
    our_pNum_avg, asd_pNum_avg = compare_avg(our_p_num_list, asd_p_num_list)


    # 计算WTL
    acc_WTL_res = compare_WTL(our_acc_list, asd_acc_list, expect = "big") # 越大越好
    asr_WTL_res = compare_WTL(our_asr_list, asd_asr_list, expect = "small") # 越小越好
    p_num_WTL_res = compare_WTL(our_p_num_list, asd_p_num_list, expect = "small") # 越小越好

    print(f"Scene:{dataset_name}|{model_name}|{attack_name}")
    print(f"OurAvg: ASR:{our_asr_avg}, ACC:{our_acc_avg}, PNUM:{our_pNum_avg}")
    print(f"ASDAvg: ASR:{asd_asr_avg}, ACC:{asd_acc_avg}, PNUM:{asd_pNum_avg}")
    print(f"WTL: ASR:{asr_WTL_res}, ACC:{acc_WTL_res}, PNUM:{p_num_WTL_res}")

if __name__ == "__main__":
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    device = torch.device("cuda:0")
    main_scene()


