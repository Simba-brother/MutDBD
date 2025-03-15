'''
# 暂时没用，未来可能要在此基础上改动
第二步:从target class中分出clean和poisoned
'''
import os
import queue
import joblib
import pandas as pd
import torch
from scipy import stats
from cliffs_delta import cliffs_delta
from codes.tools.draw import draw_box, draw_line
from codes.utils import priorityQueue_2_list, create_dir
from codes import config
from codes.common.eval_model import EvalModel
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset, ExtractTargetClassDataset
from codes.utils import entropy
# from scripts.target_class import get_adaptive_rate_of_Hybrid_mutator, get_adaptive_ratio_of_Combin_mutator
# from codes.ourMethod import TargetClassProcessor

dict_state_path = os.path.join(config.exp_root_dir, "attack", config.dataset_name, config.model_name, config.attack_name, "attack", "dict_state.pth")
dict_state = torch.load(dict_state_path, map_location="cpu")
backdoor_model = dict_state["backdoor_model"]
target_class_poisoned_set = ExtractTargetClassDataset(dict_state["purePoisonedTrainDataset"], config.target_class_idx)
target_class_clean_set = ExtractTargetClassDataset(dict_state["pureCleanTrainDataset"], config.target_class_idx)
poisoned_trainset_target_class = ExtractTargetClassDataset(dict_state["poisoned_trainset"],config.target_class_idx)
purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
device = torch.device(f"cuda:{config.gpu_id}")


def detect_poisonedAndclean_from_targetClass(
        sorted_weights_path_list,
        model_struct,
        target_class_clean_set,
        purePoisonedTrainDataset
    ):
    '''
    从target_class中分离出poisoned和clean
    Args:
        sorted_weights_path_list: 排好序的变异模型权重文件路径
        model_struct: 变异模型结构
        target_class_clean_set: target class中的clean
        purePoisonedTrainDataset: target class中的poisoned
    '''
    top_weights_list = sorted_weights_path_list[:50]

    clean_dict = {}
    poisoned_dict = {}
    for m_i, weights_path in enumerate(top_weights_list):
        weights = torch.load(weights_path, map_location="cpu")
        model_struct.load_state_dict(weights)
        e = EvalModel(model_struct, target_class_clean_set, device)
        clean_pred_labels = e._get_pred_labels()
        clean_dict[f"m_{m_i}"] = clean_pred_labels

        e = EvalModel(model_struct, purePoisonedTrainDataset, device)
        poisoned_pred_labels = e._get_pred_labels()
        poisoned_dict[f"m_{m_i}"] = poisoned_pred_labels
    df_clean = pd.DataFrame(clean_dict) # df_clean每一行是一个sample, 在这些变异模型上的预测label
    df_poisoned = pd.DataFrame(poisoned_dict)
    
    detect_q = queue.PriorityQueue()
    id = 0
    for row_id, row in df_clean.iterrows():
        pred_label_list = list(row)
        cur_instance_entropy = entropy(pred_label_list) # 熵越小越可能为poisoned,队头
        item = (cur_instance_entropy, False, id) # False => Clean, True => Poisoned
        detect_q.put(item)
        id+=1
    for row_id, row in df_poisoned.iterrows():
        pred_label_list = list(row)
        cur_instance_entropy = entropy(pred_label_list) # 熵越小越可能为poisoned,队头
        item = (cur_instance_entropy, True, id) # False => Clean, True => Poisoned
        detect_q.put(item)
        id+=1
    priority_list = priorityQueue_2_list(detect_q)
    
    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precision_list = []
    recall_list = []
    for cut_off in cut_off_list:
        end = int(len(priority_list)*cut_off)
        prefix_priority_list = priority_list[0:end]
        TP = 0
        FP = 0
        gt_TP = len(purePoisonedTrainDataset)
        for item in prefix_priority_list:
            gt_label = item[1]
            if gt_label == True:
                TP += 1
            else:
                FP += 1
        precision = round(TP/(TP+FP),3)
        recall = round(TP/gt_TP,3)
        precision_list.append(precision)
        recall_list.append(recall)
        print("cut_off:",cut_off)
        print("FP:",FP)
        print("TP:",TP)
        print("precision:",precision)
        print("recall:",recall)
        print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
        print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))
    # 绘图
    y = {"precision":precision_list, "recall":recall_list}
    title = "The relationship between detection performance and cut off"
    save_dir = os.path.join(
        config.exp_root_dir, 
        "images", 
        "line", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "entropy_seletcted_model_by_clean_seed")
    create_dir(save_dir)
    save_filename  = f"perfomance.png"
    save_path = os.path.join(save_dir, save_filename)
    x_label = "CutOff"
    draw_line(cut_off_list, title, x_label, save_path, **y)
    print(f"结果图保存在:{save_path}")
    print("detect_poisonedAndclean_from_targetClass() End")
    return priority_list, target_class_clean_set, purePoisonedTrainDataset

# def detect_poisonedAndclean_from_targetClass(
#         adaptive_mutation_rate,):
#     # 1.获得排序的变异模型
#     data_dir = os.path.join(
#         config.exp_root_dir, 
#         config.dataset_name, 
#         config.model_name, 
#         config.attack_name, 
#         "Hybrid", 
#         f"adaptive_rate_{adaptive_mutation_rate}")
#     data_file_name = "sorted_mutation_models.data"
#     data_path = os.path.join(data_dir, data_file_name)
#     priority_list = joblib.load(data_path)
#     # 取队头前50
#     top = 50
#     top_list = priority_list[:top]
#     # 前50的变异模型权重
#     top_w_file_list = []
#     for item in top_list:
#         top_w_file_list.append(item[1])
#     # 数据结构{m_i:[pred_label, pred_label]}
#     clean_dict = {}
#     poisoned_dict = {}
#     for m_i, w_file in enumerate(top_w_file_list):
#         mutation_model_state_dict = torch.load(w_file, map_location="cpu")
#         backdoor_model.load_state_dict(mutation_model_state_dict)
#         e = EvalModel(backdoor_model, target_class_clean_set, device)
#         clean_pred_labels = e._get_pred_labels()
#         clean_dict[f"m_{m_i}"] = clean_pred_labels

#         e = EvalModel(backdoor_model, purePoisonedTrainDataset, device)
#         poisoned_pred_labels = e._get_pred_labels()
#         poisoned_dict[f"m_{m_i}"] = poisoned_pred_labels
#     df_clean = pd.DataFrame(clean_dict) # df_clean每一行是一个sample, 在这些变异模型上的预测label
#     df_poisoned = pd.DataFrame(poisoned_dict)
    
#     detect_q = queue.PriorityQueue()
#     id = 0
#     for row_id, row in df_clean.iterrows():
#         id+=1
#         pred_label_list = list(row)
#         cur_instance_entropy = entropy(pred_label_list) # 熵越小越可能为poisoned,队头
#         item = (cur_instance_entropy, False, id) # False => Clean, True => Poisoned
#         detect_q.put(item)
#     for row_id, row in df_poisoned.iterrows():
#         id+=1
#         pred_label_list = list(row)
#         cur_instance_entropy = entropy(pred_label_list) # 熵越小越可能为poisoned,队头
#         item = (cur_instance_entropy, True, id) # False => Clean, True => Poisoned
#         detect_q.put(item)                                                                                      
#     priority_list = priorityQueue_2_list(detect_q)
    
#     cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#     precision_list = []
#     recall_list = []
#     for cut_off in cut_off_list:
#         end = int(len(priority_list)*cut_off)
#         prefix_priority_list = priority_list[0:end]
#         TP = 0
#         FP = 0
#         gt_TP = len(purePoisonedTrainDataset)
#         for item in prefix_priority_list:
#             gt_label = item[1]
#             if gt_label == True:
#                 TP += 1
#             else:
#                 FP += 1
#         precision = round(TP/(TP+FP),3)
#         recall = round(TP/gt_TP,3)
#         precision_list.append(precision)
#         recall_list.append(recall)
#         print("FP:",FP)
#         print("TP:",TP)
#         print("precision:",precision)
#         print("recall:",recall)
#         print("pureCleanTrainDataset num:", len(pureCleanTrainDataset))
#         print("purePoisonedTrainDataset num:", len(purePoisonedTrainDataset))
#     # 绘图
#     y = {"precision":precision_list, "recall":recall_list}
#     title = "The relationship between detection performance and cut off"
#     save_dir = os.path.join(
#         config.exp_root_dir, 
#         "images", 
#         "line", 
#         config.dataset_name, 
#         config.model_name, 
#         config.attack_name, 
#         "entropy_seletcted_model_by_clean_seed")
#     create_dir(save_dir)
#     save_filename  = f"perfomance.png"
#     save_path = os.path.join(save_dir, save_filename)
#     x_label = "CutOff"
#     draw_line(cut_off_list, title, x_label, save_path, **y)
#     print(f"结果图保存在:{save_path}")
#     print("detect_poisonedAndclean_from_targetClass() End")
#     return priority_list, target_class_clean_set, purePoisonedTrainDataset

# def get_clean_poisoned_in_target_class_of_hybrid_mutator_accuracy_list():
    '''
    所有变异算子混合 > 在不同变异率下 > 变异模型们在target_class中clean和poisoned的accuracy list
    '''
    # 返回的数据
    data_dic = {}
    # 初始化数据结构
    # {mutation_rate:{"clean":[], "poisoned":[]}}
    for mutation_rate in config.mutation_rate_list:
        data_dic[mutation_rate] = {"clean":[], "poisoned":[]}
    for mutation_name in config.mutation_name_list:
        eval_poisoned_trainset_target_class_path = os.path.join(config.exp_root_dir, config.dataset_name, config.model_name, config.attack_name, mutation_name, "eval_poisoned_trainset_target_class.data")
        eval_poisoned_trainset_target_class_report = joblib.load(eval_poisoned_trainset_target_class_path)
        for mutation_rate in config.mutation_rate_list:
            report_list = eval_poisoned_trainset_target_class_report[mutation_rate]
            for report in report_list:
                clean_acc = report["target_class_clean_acc"]
                poisoned_acc = report["target_class_poisoned_acc"]
                data_dic[mutation_rate]["clean"].append(clean_acc)
                data_dic[mutation_rate]["poisoned"].append(poisoned_acc)
    assert len(data_dic[0.01]["clean"]) == 50*len(config.model_name_list), "数量不对"

    
    return data_dic

# def analysis_clean_poisoned_in_target_class_of_Hybrid_mutator_with_adaptive_rate():
    is_dif = False
    higher = None
    dic_1,_ = get_adaptive_rate_of_Hybrid_mutator()
    adaptive_rate = dic_1["adaptive_rate"]
    dic_2 = get_clean_poisoned_in_target_class_of_hybrid_mutator_accuracy_list()
    clean_acc_list = dic_2[adaptive_rate]["clean"]
    poisoned_acc_list = dic_2[adaptive_rate]["poisoned"]
    # 计算pvalue
    p_value = stats.wilcoxon(clean_acc_list, poisoned_acc_list).pvalue
    if p_value < 0.05:
        is_dif = True
    clean_acc_list_sorted = sorted(clean_acc_list)
    poisoned_acc_list_sorted = sorted(poisoned_acc_list)
    d,info = cliffs_delta(clean_acc_list_sorted, poisoned_acc_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"

    # 绘图：
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Hybrid_clean_poisoned", "adaptive_rate")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_acc_list)
    all_y.append(poisoned_acc_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Hybrid_adaptive_rate_{adaptive_rate}"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Accuracy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{mutated_model_num*mutated_operator_num}")
    print("analysis_clean_poisoned_in_target_class() success")
    return is_dif, higher

# def get_clean_poisoned_in_target_class_with_dif_mutation_rate_by_mutation_name(mutation_name):
    '''
    得到某个变异算子下 > 在不同变异率下 > 变异模型们在各个类别上的precision
    '''
    # 返回的数据
    ans_dic = {}
    # 数据结构初始化
    # {mutation_rate:{class_idx:[precision]}}
    for mutation_rate in mutation_rate_list:
        ans_dic[mutation_rate] = {"clean":[], "poisoned":[]}
    eval_poisoned_trainset_target_class_path = os.path.join(exp_root_dir,dataset_name,model_name,attack_name, mutation_name, "eval_poisoned_trainset_target_class.data")
    eval_poisoned_trainset_target_class_report = joblib.load(eval_poisoned_trainset_target_class_path)
    for mutation_rate in mutation_rate_list:
        report_list = eval_poisoned_trainset_target_class_report[mutation_rate]
        for report in report_list:
            clean_acc = report["target_class_clean_acc"]
            poisoned_acc = report["target_class_poisoned_acc"]
            ans_dic[mutation_rate]["clean"].append(clean_acc)
            ans_dic[mutation_rate]["poisoned"].append(poisoned_acc)
    assert len(ans_dic[0.01]["clean"]) == mutated_model_num, "数量不对"
    return ans_dic

# def get_clean_poisoned_acc_list_in_target_class_of_combin_mutator():
    dic = get_adaptive_ratio_of_Combin_mutator()
    clean_acc_list = []
    poisoned_acc_list = []
    for mutation_name in config.mutation_name_list:
        adaptive_rate = dic[mutation_name]["adaptive_rate"]
        if adaptive_rate == -1:
            continue
        dic_1 = get_clean_poisoned_in_target_class_with_dif_mutation_rate_by_mutation_name(mutation_name)
        temp_clean_acc_list = dic_1[adaptive_rate]["clean"]
        temp_poisoned_acc_list = dic_1[adaptive_rate]["poisoned"]
        clean_acc_list.extend(temp_clean_acc_list)
        poisoned_acc_list.extend(temp_poisoned_acc_list)
    print("cur_mutated_model_num:",len(clean_acc_list)) 
    return clean_acc_list, poisoned_acc_list

# def analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate():
    is_dif = False
    higher = None
    clean_acc_list, poisoned_acc_list = get_clean_poisoned_acc_list_in_target_class_of_combin_mutator()
    # 计算pvalue
    if clean_acc_list == poisoned_acc_list:
        p_value = float("inf")
    else:
        p_value = stats.wilcoxon(clean_acc_list, poisoned_acc_list).pvalue
    if p_value < 0.05:
        is_dif = True
    clean_acc_list_sorted = sorted(clean_acc_list)
    poisoned_acc_list_sorted = sorted(poisoned_acc_list)
    d,info = cliffs_delta(clean_acc_list_sorted, poisoned_acc_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"
    # 绘图：
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Combin_clean_poisoned", "adaptive_rate")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_acc_list)
    all_y.append(poisoned_acc_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Combin_adaptive_rate"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Accuracy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{len(clean_acc_list)}")
    print("analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate() success")
    return is_dif, higher

# def clean_poisoned_acc_dif():
    temp_dic, _ = get_adaptive_rate_of_Hybrid_mutator()
    mutation_rate = temp_dic['adaptive_rate']
    data_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, "Hybrid", f"adaptive_rate_{mutation_rate}")
    data_file_name = "sorted_mutation_models.data"
    data_path = os.path.join(data_dir, data_file_name)
    priority_list = joblib.load(data_path)
    top = 50
    top_list = priority_list[:top]
    top_w_file_list = []
    for item in top_list:
        top_w_file_list.append(item[1])
    poisoned_acc_list = []
    clean_acc_list = []
    for w_file in top_w_file_list:
        mutation_model_state_dict = torch.load(w_file, map_location="cpu")
        backdoor_model.load_state_dict(mutation_model_state_dict)
        e = EvalModel(backdoor_model, target_class_poisoned_set, device)
        poisoned_acc = e._eval_acc()
        e = EvalModel(backdoor_model, target_class_clean_set, device)
        clean_acc = e._eval_acc()
        poisoned_acc_list.append(poisoned_acc)
        clean_acc_list.append(clean_acc)

    is_dif = False
    higher = None
    # 计算pvalue
    if clean_acc_list == poisoned_acc_list:
        p_value = float("inf")
    else:
        p_value = stats.wilcoxon(clean_acc_list, poisoned_acc_list).pvalue
    if p_value < 0.05:
        is_dif = True

    clean_acc_list_sorted = sorted(clean_acc_list)
    poisoned_acc_list_sorted = sorted(poisoned_acc_list)
    d,info = cliffs_delta(clean_acc_list_sorted, poisoned_acc_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"
    
    # 绘图
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Hybrid_clean_poisoned_select_model", "adaptive_rate")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_acc_list)
    all_y.append(poisoned_acc_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Hybrid_adaptive_rate_selected_model"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Accuracy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{len(clean_acc_list)}")
    print("clean_poisoned() success")
    return is_dif, higher


# def clean_poisoned_entropy_dif():
    temp_dic, _ = get_adaptive_rate_of_Hybrid_mutator()
    mutation_rate = temp_dic['adaptive_rate']
    data_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, "Hybrid", f"adaptive_rate_{mutation_rate}")
    data_file_name = "sorted_mutation_models.data"
    data_path = os.path.join(data_dir, data_file_name)
    priority_list = joblib.load(data_path)
    top = 50
    top_list = priority_list[:top]
    top_w_file_list = []
    for item in top_list:
        top_w_file_list.append(item[1])
    poisoned_entropy_list = []
    clean_entropy_list = []
    for w_file in top_w_file_list:
        mutation_model_state_dict = torch.load(w_file, map_location="cpu")
        backdoor_model.load_state_dict(mutation_model_state_dict)
        e = EvalModel(backdoor_model, target_class_poisoned_set, device)
        poisoned_pred_labels = e._get_pred_labels()
        poisoned_entropy = entropy(poisoned_pred_labels)
        e = EvalModel(backdoor_model, target_class_clean_set, device)
        clean_pred_labels = e._get_pred_labels()
        clean_entropy = entropy(clean_pred_labels)
        poisoned_entropy_list.append(poisoned_entropy)
        clean_entropy_list.append(clean_entropy)

    is_dif = False
    higher = None
    # 计算pvalue
    if clean_entropy_list == poisoned_entropy_list:
        p_value = float("inf")
    else:
        p_value = stats.wilcoxon(clean_entropy_list, poisoned_entropy_list).pvalue
    if p_value < 0.05:
        is_dif = True

    clean_entropy_list_sorted = sorted(clean_entropy_list)
    poisoned_entropy_list_sorted = sorted(poisoned_entropy_list)
    d,info = cliffs_delta(clean_entropy_list_sorted, poisoned_entropy_list_sorted)
    if d < 0:
        higher = "poisoned"
    else:
        higher = "clean"
    
    # 绘图
    save_dir = os.path.join(exp_root_dir, "images/box", dataset_name, model_name, attack_name, "Hybrid_clean_poisoned_select_model", "adaptive_rate", "entropy")
    create_dir(save_dir)
    all_y = []
    labels = []
    all_y.append(clean_entropy_list)
    all_y.append(poisoned_entropy_list)
    labels.append("Clean")
    labels.append("Poisoned")
    title = f"{dataset_name}_{model_name}_{attack_name}_Hybrid_adaptive_rate_selected_model"
    save_file_name = title+".png"
    save_path = os.path.join(save_dir, save_file_name)
    xlabel = "TargetClass"
    ylabel = "Entropy"
    draw_box(all_y, labels, title, xlabel, ylabel, save_path)
    print(f"mutated_model_num:{len(clean_entropy_list)}")
    print("clean_poisoned_entropy_dif() success")
    return is_dif, higher

if __name__ == "__main__":

    # detect_poisonedAndclean_from_targetClass()
    pass

    # is_dif, higher = analysis_clean_poisoned_in_target_class_of_Hybrid_mutator_with_adaptive_rate()
    # print("==================")
    # print(f"dataset:{dataset_name}")
    # print(f"model:{model_name}")
    # print(f"attack:{attack_name}")
    # print(f"is_dif:{is_dif}")
    # print(f"higher:{higher}")

    # is_dif, higher = clean_poisoned()
    # print("==================")
    # print(f"dataset:{dataset_name}")
    # print(f"model:{model_name}")
    # print(f"attack:{attack_name}")
    # print(f"is_dif:{is_dif}")
    # print(f"higher:{higher}")

   

    # is_dif, higher = analysis_clean_poisoned_in_target_class_of_Combin_mutator_with_adaptive_rate()
    # print("==================")
    # print(f"dataset:{dataset_name}")
    # print(f"model:{model_name}")
    # print(f"attack:{attack_name}")
    # print(f"is_dif:{is_dif}")
    # print(f"higher:{higher}")