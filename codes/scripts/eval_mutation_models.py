'''
对变异模型进行评估的脚本
'''
import sys
sys.path.append("./")
import random
import os
import numpy as np
import queue
import torch
from tqdm import tqdm
import setproctitle
import joblib
from collections import defaultdict

from codes.modelMutat import ModelMutat_2
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractTargetClassDataset, ExtractDatasetByIds, CombinDataset

from codes.utils import create_dir, priorityQueue_2_list
from codes import config
from codes.eval_model import EvalModel
from codes import draw
from codes.scripts.baseData import BaseData

from codes.scripts.target_class import TargetClassProcessor

# 获得配置信息
mutation_rate_list = config.mutation_rate_list
exp_root_dir = config.exp_root_dir
dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_operator_name_list = config.mutation_name_list
class_num = config.class_num
# 随机种子
random.seed(666)
# 获得攻击成功的状态信息,并加载到cpu上
dict_state_path = os.path.join(exp_root_dir,"attack",dataset_name,model_name,attack_name,"attack","dict_state.pth")
dict_state = torch.load(dict_state_path, map_location="cpu")
# 获得后门模型
backdoor_model = dict_state["backdoor_model"]
mutation_num = config.mutation_model_num
target_class_idx = config.target_class_idx 
# GPU设备
device = torch.device("cuda:1")


# def eval_weight_gf_mutated_models_in_target_class():
#     '''
#     dataset/model_name/attack_name/weight_gf
#     评估在后门训练集上target class上(clean,posioned,整体)的accuracy
#     '''
#     mutation_operator_name = "weight_gf"
#     save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
#     create_dir(save_dir)
#     save_file_name = f"eval_poisoned_trainset_target_class.data"
#     save_path =  os.path.join(save_dir, save_file_name)
#     # 目标类索引
#     target_class_idx = 1
#     # 把目标类数据集分为clean和poisoned
#     target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], target_class_idx)
#     target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], target_class_idx)
#     # whole_target_set = ExtractTargetClassDataset(dict_state["poisoned_trainset"], target_class_idx)
#     # {mutation_ratio:[{"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole}]}
#     res_dict = dict()
#     for mutation_rate in tqdm(config.fine_mutation_rate_list):
#         mutation_models_dir = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate)) 
#         # 获得该变异率下的变异模型的state_dict
#         temp_list = []
#         for m_i in range(mutation_num):
#             mutation_model_state_dict_path = os.path.join(mutation_models_dir, f"mutated_model_{m_i}.pth")
#             mutation_model_state_dict = torch.load(mutation_model_state_dict_path, map_location="cpu")
#             backdoor_model.load_state_dict(mutation_model_state_dict)
#             e = EvalModel(backdoor_model, target_class_clean_set, device)
#             acc_clean = e._eval_acc()
#             e = EvalModel(backdoor_model, target_class_poisoned_set, device)
#             acc_poisoned = e._eval_acc()
#             # e = EvalModel(backdoor_model, whole_target_set, device)
#             # acc_whole = e._eval_acc()
#             temp_list.append({"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned})
#         res_dict[mutation_rate] = temp_list
#     joblib.dump(res_dict, save_path)
#     # 整理数据
#     data = joblib.load(save_path)
#     mean_p_acc_list = []
#     mean_c_acc_list = []
#     for mutation_rate in config.fine_mutation_rate_list:
#         p_acc_list = []
#         c_acc_list = []
#         for item in data[mutation_rate]:
#             clean_acc = item["target_class_clean_acc"]
#             poisoned_acc = item["target_class_poisoned_acc"]
#             c_acc_list.append(clean_acc)
#             p_acc_list.append(poisoned_acc)
#         mean_p = sum(p_acc_list)/len(p_acc_list)
#         mean_c = sum(c_acc_list)/len(c_acc_list)
#         mean_p_acc_list.append(mean_p)
#         mean_c_acc_list.append(mean_c)
#     # 画图
#     x_ticks = config.fine_mutation_rate_list
#     title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
#     xlabel = "mutation_rate"
#     save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
#     create_dir(save_dir)
#     save_file_name = "targetclass_clean_poisoned_accuracy_variation"
#     save_path = os.path.join(save_dir, save_file_name)
#     y = {"poisoned set":mean_p_acc_list, "clean set":mean_c_acc_list}
#     draw.draw_line(x_ticks, title, xlabel, save_path, **y)
#     print("draw_eval_mutated_model_in_target_class() successful")
#     return res_dict


def eval_mutated_model(mutation_operator_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在整个后门训练集(poisoned trainset)上各个分类上的report
    '''
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = f"eval_poisoned_trainset_report.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 整个后门训练集
    poisoned_trainset = dict_state["poisoned_trainset"]
    report_data = defaultdict(list)
    # 遍历变异率
    for mutation_rate in tqdm(mutation_rate_list):
        # 获得该变异率的变异模型目录
        mutation_models_dir = os.path.join(
            exp_root_dir, 
            "mutations", 
            dataset_name, 
            model_name, 
            attack_name, 
            mutation_operator_name, 
            str(mutation_rate))
        # 遍历变异模型
        for m_i in range(mutation_num):
            state_dict = torch.load(os.path.join(
                mutation_models_dir, 
                f"mutated_model_{m_i}.pth"), 
                map_location="cpu")
            backdoor_model.load_state_dict(state_dict)
            evalModel = EvalModel(backdoor_model, poisoned_trainset, device)
            report = evalModel._eval_classes_acc()
            report_data[mutation_rate].append(report)
    joblib.dump(report_data, save_path)
    # 整理数据
    # 存储各个变异率下poisoned_trainset上的mean accuracy
    mean_acc_list = []
    for mutation_rate in mutation_rate_list:
        acc_list = []
        report_list = report_data[mutation_rate]
        for report in report_list:
            acc_list.append(report["accuracy"]) 
        mean_acc = sum(acc_list)/len(acc_list)
        mean_acc_list.append(mean_acc)
    # 画图
    x_ticks = mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = "Accuracy varies with mutation rate"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"poisoned_trainset":mean_acc_list}
    draw.draw_line(x_ticks, title, xlabel, save_path, **y)
    print("eval_mutated_model() successful")
    # 返回数据
    return report_data

def eval_mutated_model_in_target_class(mutation_operator_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在后门训练集上target class上(clean,posioned,整体)的accuracy
    '''
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = f"eval_poisoned_trainset_target_class.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 把目标类数据集分为clean和poisoned
    target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], target_class_idx)
    target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], target_class_idx)
    # whole_target_set = ExtractTargetClassDataset(dict_state["poisoned_trainset"], target_class_idx)
    # 存储的数据结构
    # {mutation_ratio:[{"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole}]}
    res_dict = dict()
    # 遍历变异率
    for mutation_rate in tqdm(mutation_rate_list):
        mutation_models_dir = os.path.join(
            exp_root_dir, 
            "mutations", 
            dataset_name, 
            model_name, 
            attack_name, 
            mutation_operator_name, 
            str(mutation_rate)) 
        # 用于存储该变异率下变异模型list的评估结果
        temp_list = []
        for m_i in range(mutation_num):
            mutation_model_state_dict_path = os.path.join(mutation_models_dir, f"mutated_model_{m_i}.pth")
            mutation_model_state_dict = torch.load(mutation_model_state_dict_path, map_location="cpu")
            backdoor_model.load_state_dict(mutation_model_state_dict)
            e = EvalModel(backdoor_model, target_class_clean_set, device)
            acc_clean = e._eval_acc()
            e = EvalModel(backdoor_model, target_class_poisoned_set, device)
            acc_poisoned = e._eval_acc()
            # e = EvalModel(backdoor_model, whole_target_set, device)
            # acc_whole = e._eval_acc()
            temp_list.append({"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned})
        res_dict[mutation_rate] = temp_list
    joblib.dump(res_dict, save_path)
    # 整理数据
    data = joblib.load(save_path)
    mean_p_acc_list = []
    mean_c_acc_list = []
    for mutation_rate in  mutation_rate_list:
        p_acc_list = []
        c_acc_list = []
        for item in data[mutation_rate]:
            clean_acc = item["target_class_clean_acc"]
            poisoned_acc = item["target_class_poisoned_acc"]
            c_acc_list.append(clean_acc)
            p_acc_list.append(poisoned_acc)
        mean_p = sum(p_acc_list)/len(p_acc_list)
        mean_c = sum(c_acc_list)/len(c_acc_list)
        mean_p_acc_list.append(mean_p)
        mean_c_acc_list.append(mean_c)
    # 画图
    x_ticks = mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
    create_dir(save_dir)
    save_file_name = "targetclass_clean_poisoned_accuracy_variation"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"poisoned set":mean_p_acc_list, "clean set":mean_c_acc_list}
    draw.draw_line(x_ticks, title, xlabel, save_path, **y)
    print("draw_eval_mutated_model_in_target_class() successful")
    return res_dict

def eval_mutated_model_all_mutation_operator():
    ans_dict = defaultdict(list)
    # 遍历变异算子
    for mutation_operator_name in mutation_operator_name_list:
        eval_poisoned_trainset_report = joblib.load(
            os.path.join(
                exp_root_dir, 
                dataset_name, 
                model_name, 
                attack_name, 
                mutation_operator_name, 
                "eval_poisoned_trainset_report.data"))
        # 遍历变异率
        for mutation_rate in  mutation_rate_list:
            report_list = eval_poisoned_trainset_report[mutation_rate]
            for report in report_list:
                ans_dict[mutation_rate].append(report["accuracy"])
    assert len(ans_dict[0.01]) == mutation_num*len(mutation_operator_name_list)
    # 画图
    mean_acc_list = []
    for mutation_rate in  mutation_rate_list:
        mean_acc_list.append(np.mean(ans_dict[mutation_rate]))
    x_ticks = mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:All"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, "All")
    create_dir(save_dir)
    save_file_name = "Accuracy varies with mutation rate"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"poisoned_trainset":mean_acc_list}
    draw.draw_line(x_ticks, title, xlabel, save_path, **y)
    print("eval_mutated_model_all_mutation_operator() successful")

def eval_mutated_model_in_target_class_all_mutation_operator():
    clean_dict = defaultdict(list)
    poisoned_dict = defaultdict(list)
    for mutation_operator_name in mutation_operator_name_list:
        eval_poisoned_trainset_target_class_report = joblib.load(os.path.join(
            exp_root_dir, 
            dataset_name, 
            model_name, 
            attack_name, 
            mutation_operator_name, 
            "eval_poisoned_trainset_target_class.data"))
        for mutation_rate in  mutation_rate_list:
            report_list = eval_poisoned_trainset_target_class_report[mutation_rate]
            for report in report_list:
                clean_dict[mutation_rate].append(report["target_class_clean_acc"])
                poisoned_dict[mutation_rate].append(report["target_class_poisoned_acc"])
    assert len(clean_dict[0.01]) == mutation_num*5, "数量不对"
    assert len(poisoned_dict[0.01]) == mutation_num*5, "数量不对"
    # 画图
    mean_clean_acc_list = []
    mean_poisoned_acc_list = []
    for mutation_rate in  mutation_rate_list:
        mean_clean_acc_list.append(np.mean(clean_dict[mutation_rate]))
        mean_poisoned_acc_list.append(np.mean(poisoned_dict[mutation_rate]))
    x_ticks = mutation_rate_list
    title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:All"
    xlabel = "mutation_rate"
    save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, "All")
    create_dir(save_dir)
    save_file_name = "targetclass_clean_poisoned_accuracy_variation"
    save_path = os.path.join(save_dir, save_file_name)
    y = {"poisoned set":mean_poisoned_acc_list, "clean set":mean_clean_acc_list}
    draw.draw_line(x_ticks, title, xlabel, save_path, **y)
    print("eval_mutated_model_in_target_class_all_mutation_operator() successful")

# def draw_eval_mutated_model_in_target_class(mutation_operator_name):
#     # 整理数据
#     save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, mutation_operator_name)
#     data_filename = "eval_poisoned_trainset_target_class.data"
#     data = joblib.load(os.path.join(save_dir, data_filename))
#     mean_p_acc_list = []
#     mean_c_acc_list = []
#     for mutation_rate in  mutation_rate_list:
#         p_acc_list = []
#         c_acc_list = []
#         for item in data[mutation_rate]:
#             clean_acc = item["target_class_clean_acc"]
#             poisoned_acc = item["target_class_poisoned_acc"]
#             c_acc_list.append(clean_acc)
#             p_acc_list.append(poisoned_acc)
#         mean_p = sum(p_acc_list)/len(p_acc_list)
#         mean_c = sum(c_acc_list)/len(c_acc_list)
#         mean_p_acc_list.append(mean_p)
#         mean_c_acc_list.append(mean_c)
#     # 画图
#     x_ticks = mutation_rate_list
#     title = f"Dataset:{dataset_name}, Model:{model_name}, attack_name:{attack_name}, mutation_operator_name:{mutation_operator_name}"
#     xlabel = "mutation_rate"
#     save_dir = os.path.join(exp_root_dir,"images/line", dataset_name, model_name, attack_name, mutation_operator_name)
#     create_dir(save_dir)
#     save_file_name = "targetclass_clean_poisoned_accuracy_variation"
#     save_path = os.path.join(save_dir, save_file_name)
#     y = {"poisoned set":mean_p_acc_list, "clean set":mean_c_acc_list}
#     draw.draw_line(x_ticks, title, xlabel, save_path, **y)
#     print("draw_eval_mutated_model_in_target_class() successful")
    
def sort_mutated_model():
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    # target class中的clean set
    target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], target_class_idx)
    # target class中的poisoned set
    target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], target_class_idx)
    # 每个类别选10个clean samples
    seed_num = 10*class_num
    # clean set ids
    ids = list(range(len(target_class_clean_set)))
    # 打乱编号顺序
    random.shuffle(ids)
    # 选择seed_ids
    selected_ids = ids[0:seed_num]
    # 获得remain_ids
    remain_ids = list(set(ids) - set(selected_ids))
    clean_seed_dataset = ExtractDatasetByIds(target_class_clean_set, selected_ids)
    clean_remain_dataset = ExtractDatasetByIds(target_class_clean_set, remain_ids)
    remain_dataset = CombinDataset(clean_remain_dataset,target_class_poisoned_set)
    # 用于存储所有变异算子自适应出的变异率下的变异模型权重
    weight_file_list = []
    for mutation_operator_name in mutation_operator_name_list:
        baseData = BaseData(dataset_name, model_name, attack_name, mutation_operator_name)
        targetClassProcessor = TargetClassProcessor(
            dataset_name,
            model_name, # 模型名称
            attack_name, # 攻击名称
            mutation_operator_name_list, # 变异算子名称list
            mutation_rate_list, # 变异率list
            exp_root_dir, # 实验数据根目录
            class_num, # 数据集的分类数
            mutated_model_num = config.mutation_model_num,
            mutation_operator_num = len(mutation_operator_name_list)
        )
        temp_dic, _ = targetClassProcessor.get_adaptive_rate_of_Hybrid_mutator()
        # 自适应的变异率
        mutation_rate = temp_dic['adaptive_rate']
        weight_file_list.extend(baseData.get_mutation_weight_file_by_mutation_rate(mutation_rate))
    for weight_file in weight_file_list:
        state_dict = torch.load(weight_file, map_location="cpu")
        backdoor_model.load_state_dict(state_dict)
        e = EvalModel(backdoor_model, clean_seed_dataset, device)
        acc_seed = e._eval_acc()
        e = EvalModel(backdoor_model, remain_dataset, device)
        acc_remain = e._eval_acc()
        priority = acc_seed - acc_remain # 越小,优先级越高,因为越小有可能acc_seed很低，说明该变异模型在clean seed上预测的很乱。同时acc_remain很高，说明该变异模型在poisoned seed上预测的很稳定
        item = (priority, weight_file)
        q.put(item)
    priority_list = priorityQueue_2_list(q)
    save_dir = os.path.join(
        exp_root_dir, 
        dataset_name, 
        model_name, 
        attack_name, 
        "Hybrid", 
        f"adaptive_rate_{mutation_rate}")
    create_dir(save_dir)
    save_file_name = "sorted_mutation_models.data"
    save_path = os.path.join(save_dir, save_file_name)
    joblib.dump(priority_list, save_path)
    print("sort_mutated_model() success")
    return priority_list


 
if __name__ == "__main__":

    # setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval_mutated_models")
    # for mutation_operator in mutation_operator_name_list[1:]:
    #     # mutation_operator = "gf"
    #     print(f"mutation_operator:{mutation_operator}")
    #     eval_mutated_model(mutation_operator_name=mutation_operator)

    # setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval_target_class")
    # print(dataset_name+"_"+attack_name+"_"+model_name+"_eval_target_class")
    # for mutation_operator in mutation_operator_name_list:
    #     print(f"mutation_operator:{mutation_operator}")
    #     eval_mutated_model_in_target_class(mutation_operator_name=mutation_operator)

    # eval_mutated_model_all_mutation_operator()

    # eval_mutated_model_in_target_class_all_mutation_operator()

    # setproctitle.setproctitle(dataset_name+"_"+attack_name+"_"+model_name+"_eval_weight_gf_mutated_models")
    # eval_weight_gf_mutated_models_in_target_class()

    sort_mutated_model()
    pass