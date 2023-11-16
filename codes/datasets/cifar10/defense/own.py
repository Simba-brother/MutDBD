import sys
sys.path.append("./") 
import copy
from collections import defaultdict
import random
import os
import joblib

import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader,Dataset


# from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_BadNets_dict_state
# from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Blended_dict_state
# from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import PoisonedTrainDataset, PurePoisonedTrainDataset, PureCleanTrainDataset, PoisonedTestSet, TargetClassCleanTrainDataset,  get_dict_state as get_IAD_dict_state
# from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_LabelConsist_dict_state
from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Refool_dict_state
# from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_WaNet_dict_state


from codes.utils import entropy
# 从攻击脚本中获得配置
origin_dict_state = get_Refool_dict_state()
# 本脚本全局变量
# 后门模型
backdoor_model = origin_dict_state["backdoor_model"]
poisoned_trainset = origin_dict_state["poisoned_trainset"]
purePoisonedTrainDataset = origin_dict_state["purePoisonedTrainDataset"]
pureCleanTrainDataset = origin_dict_state["pureCleanTrainDataset"]
target_label = 1

class PureTargetClassCleanTrainset(Dataset):
    def __init__(self, pureCleanTrainDataset, target_label):
        self.pureCleanTrainDataset = pureCleanTrainDataset
        self.target_label  = target_label
        self.pureTargetClassCleanTrainset = self._get_pureTargetClassCleanTrainset()

    def _get_pureTargetClassCleanTrainset(self):
        pureTargetClassCleanTrainset = []
        for id in range(len(self.pureCleanTrainDataset)):
            sample, label = self.pureCleanTrainDataset[id]
            if label == self.target_label:
                pureTargetClassCleanTrainset.append((sample,label))
        return pureTargetClassCleanTrainset
    
    def __len__(self):
        return len(self.pureTargetClassCleanTrainset)
    
    def __getitem__(self, index):
        x,y=self.pureTargetClassCleanTrainset[index]
        return x,y
pureTargetClassCleanTrainset = PureTargetClassCleanTrainset(pureCleanTrainDataset, target_label)
# 变异模型数量
mutated_model_num =10
# 变异模型权重存储目录
mutated_model_dir = "experiments/CIFAR10/resnet18_nopretrain_32_32_3/mutates/gf/Refool"
# 脚本工作目录
work_dir = "experiments/CIFAR10/resnet18_nopretrain_32_32_3/defense/gf/Refool"

# 设备
device = torch.device('cuda:1')

def _seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def identify_target_label():
    '''
    识别攻击者的target_label
    '''
    m_o_dic = defaultdict(float) # key:class_idx, val: acc 
    m_m_dic = defaultdict(list) # key:class_idx, val: list: [acc_0,acc_1...,acc_9]
    dif_dic = defaultdict(float) # key:class_idx, val: mean acc dif
    print(poisoned_trainset.class_to_idx)
    # 获得模型结构
    model_o = backdoor_model
    # 深度拷贝
    model_m = copy.deepcopy(model_o)
    # model_o准备预测
    correct_predictions = defaultdict(int) # key: class_idx, value: 该类别正确数目
    total_predictions = defaultdict(int) # key: class_idx, value: 该类别总共数目
    # 开始分类统计model_o精度
    batch_size =128
    poisoned_trainset_loader = DataLoader(
        poisoned_trainset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    model_o.eval() 
    model_o.to(device)
    with torch.no_grad():
        for X, Y in poisoned_trainset_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model_o(X)
            predictions = torch.argmax(outputs, dim=1)
            for i in range(len(Y)):
                y = Y[i].item()
                p_y = predictions[i].item()
                total_predictions[y] += 1
                if y == p_y:
                    correct_predictions[y] += 1

    for class_idx in range(10):
        correct_num = correct_predictions[class_idx]
        total_num = total_predictions[class_idx]
        acc = round(correct_num/total_num, 3)
        m_o_dic[class_idx] = acc


    # 开始分类统计model_m精度
    print("开始分类统计model_m精度")
    with torch.no_grad():
        for m_i in range(10): # 使用10个 mutated model
            # 计算第i个变异模型的分类精度
            model_m.load_state_dict(torch.load(os.path.join(mutated_model_dir, f"model_mutated_{m_i+1}.pth")))
            model_m.eval()
            model_m.to(device)
            correct_predictions = defaultdict(int) # key: class_idx, value: 该类别正确数目
            total_predictions = defaultdict(int) # key: class_idx, value: 该类别总共数目
            for X, Y in poisoned_trainset_loader:
                X = X.to(device)
                Y = Y.to(device)
                outputs = model_m(X)
                predictions = torch.argmax(outputs, dim=1)
                for i in range(len(Y)):
                    y = Y[i].item()
                    p_y = predictions[i].item()
                    total_predictions[y] += 1
                    if y == p_y:
                        correct_predictions[y] += 1
            for class_idx in range(10):
                correct_num = correct_predictions[class_idx]
                total_num = total_predictions[class_idx]
                acc = round(correct_num/total_num, 3)
                m_m_dic[class_idx].append(acc)
        
    for class_idx in range(10):
        m_o_acc = m_o_dic[class_idx]
        dif_sum = 0
        dif_mean = 0
        for m_acc in m_m_dic[class_idx]:
            dif_sum += m_o_acc - m_acc
        dif_mean = round(dif_sum / len(m_m_dic[class_idx]),3)
        dif_dic[class_idx] = dif_mean
    print(dif_dic)
    min_mean_dif = float('inf')
    target_y = None
    for class_idx, mean_dif in dif_dic.items():
        if mean_dif < min_mean_dif:
            min_mean_dif = mean_dif
            target_y = class_idx
    print("target_y:", target_y)
    return target_y

    '''
    统计clean samples 和 poisoned samples 在 mutated models上的熵
    2: 对于每个clean sample计算一下在mutated models上的熵 然后加起来算平均。同理poisoned sample。理论上这种poisoned熵会更小。

    # 得到模型结构
    config = get_wanet_config()
    model_o = config["backdoor_model"]
    # 得到攻击class_idx
    target_class_idx = config["target_class_idx"]
    # 得到变异模型结构
    model_m = copy.deepcopy(model_o)
    # 得到中毒训练集
    poisoned_trainset = config["poisoned_trainset"]
    # 得到样本总数
    total_sample_num = len(poisoned_trainset)
    # 得到污染样本ids
    poisoned_ids = poisoned_trainset.poisoned_set
    # 获得设备
    device = torch.device('cuda:4')
    # 批次大小
    batch_size = 128
    # 中毒集加载器！！不要打乱
    poisoned_trainset_loader = DataLoader(
        poisoned_trainset,
        batch_size = batch_size,
        shuffle=False, ## importent
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    # 变异模型存储目录
    mutated_models_dir = "experiments/CIFAR10/models/resnet18_nopretrain_32_32_3/mutates/wanet"
    # 存储变异模型对所有样本的预测标签
    predict_dic = defaultdict(list) # predict_dic[m_i][sample_id]: m_i对sample_id的预测标签
    for m_i in range(10):
        predict_dic[m_i] = [None]*total_sample_num
    # 跟踪批次中样本的id
    sample_id = 0
    with torch.no_grad():
        for X, Y in poisoned_trainset_loader:
            # 拿到一个批次数据
            X = X.to(device)
            Y = Y.to(device)
            sample_ids= [None]*len(Y) # 这一批次样本的全局id sample id
            for i in range(len(Y)):
                sample_ids[i] = sample_id
                sample_id += 1 # id ++
            # 在该批次中遍历m_i
            for m_i in range(10):
                # 拿到1个变异模型
                mutated_weight_path = os.path.join(mutated_models_dir, f"model_mutated_{m_i+1}.pth")
                model_m.load_state_dict(torch.load(mutated_weight_path, map_location="cpu"))
                model_m.eval()
                model_m.to(device)
                outputs = model_m(X)
                predictions = torch.argmax(outputs, dim=1)
                for i in range(len(Y)):
                    sample_id = sample_ids[i]
                    p_y = predictions[i].item()
                    predict_dic[m_i][sample_id] = p_y
    joblib.dump(predict_dic,"cifar_10_resnet18_nopretrain_32_32_wanet_mutate_model_predict_on_poisoned_trainset_dic.data")
    # 获得在target class idx中数据集中的污染样本ID
    poisoned_sample_ids = []
    # 获得在target class idx中数据集中的干净样本ID
    clean_sample_ids = []
    # 获得在target class idx中数据集中样本ID
    target_sample_ids = []
    # 遍历数据集
    for i in range(len(poisoned_trainset)):
        x, y = poisoned_trainset[i]
        if y == target_class_idx:
            target_sample_ids.append(i)
            if i in poisoned_ids:
                poisoned_sample_ids.append(i)
            else:
                clean_sample_ids.append(i)
    assert len(poisoned_sample_ids) == len(poisoned_ids), "目标类中污染集应该就是全部的污染集"
    # 存储target class idx中数据集中污染样本的熵
    poisoned_entroy_list = []
    # 存储target class idx中数据集中干净样本的熵
    clean_entroy_list = []
    # 存储target class idx中数据集中样本的熵
    target_entroy_list = []
    for poisoned_sample_id in poisoned_sample_ids:
        predict_label = []
        for m_i in predict_dic.keys():
            predict_label.append(predict_dic[m_i][poisoned_sample_id])
        poisoned_entroy_list.append(entropy(predict_label))
    for clean_sample_id in clean_sample_ids:
        predict_label = []
        for m_i in predict_dic.keys():
            predict_label.append(predict_dic[m_i][clean_sample_id])
        clean_entroy_list.append(entropy(predict_label))

    for target_sample_id in target_sample_ids:
        predict_label = []
        for m_i in predict_dic.keys():
            predict_label.append(predict_dic[m_i][target_sample_id])
        target_entroy_list.append(entropy(predict_label))

    poisoned_mean_entroy = np.mean(poisoned_entroy_list)
    clean_mean_entroy = np.mean(clean_entroy_list)
    print(f"poisoned mean entroy: {poisoned_mean_entroy},  clean mean entroy: {clean_mean_entroy}")
    
    # 检测
    threshold = clean_mean_entroy
    ground_truth = []
    detect_list = []
    for target_sample_id in target_sample_ids:
        if target_sample_id in poisoned_ids:
            ground_truth.append(True)
        else:
            ground_truth.append(False)
        if target_entroy_list[target_sample_id] < threshold:
            detect_list.append(True)
        else:
            detect_list.append(False)
    
    TN, FP, FN, TP = confusion_matrix(ground_truth, detect_list).ravel()
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1score = 2 * Precision * Recall /(Precision + Recall)
    print("TN",TN)
    print("FP",FP)
    print("FN",FN)
    print("TP",TP)
    print("Precision",Precision)
    print("Recall", Recall)
    print("F1score", F1score)
    '''
def get_mutated_model_predict_label(dataset):
    dataset_loader = DataLoader(
        dataset,
        batch_size = 128,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    model = backdoor_model
    mutated_models_predict_label_dic= defaultdict(list)
    mutated_models_acc = []
    for m_i in range(mutated_model_num):
        # 加载变异权重
        state_dict = torch.load(os.path.join(mutated_model_dir, f"model_mutated_{m_i+1}.pth"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        correct_num = 0
        total_num = len(dataset)
        for X,Y in dataset_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            predict_label = torch.argmax(outputs, dim=1).tolist()
            mutated_models_predict_label_dic[f"m_{m_i}"].extend(predict_label)
            correct_num += (torch.argmax(outputs, dim=1) == Y).sum()
        acc = round(correct_num.item()/total_num,3)
        mutated_models_acc.append(acc)
    return mutated_models_acc, mutated_models_predict_label_dic

def detect(threshold):
    poisoned_df = pd.read_csv(os.path.join(work_dir,"poisoned_mutated_models_predict_label_dic.csv"))
    clean_df = pd.read_csv(os.path.join(work_dir,"clean_mutated_models_predict_label_dic.csv"))
    gt_list = []
    predict_list = []
    for row_id, row in poisoned_df.iterrows():
        gt_list.append(True)
        if entropy(list(row)) <  threshold:
            predict_list.append(True)
        else:
            predict_list.append(False)
    for row_id, row in clean_df.iterrows():
        gt_list.append(False)
        if entropy(list(row)) <  threshold:
            predict_list.append(True)
        else:
            predict_list.append(False)
    TN, FP, FN, TP = confusion_matrix(gt_list, predict_list).ravel()
    acc = round((TP+TN)/(TP+TN+FP+FN),3)
    precision = round(TP/(TP+FP),3)
    recall = round(TP/(TP+FN),3)
    F1 = round(2*precision*recall/(precision+recall),3)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)
    print("TP:",TP)
    print("acc:",acc)
    print("precision:",precision)
    print("recall:",recall)
    print("F1:",F1)

def caclue_entropy(df):
    mean_entroy = 0
    temp_sum = 0
    count = 0
    for row_id, row in df.iterrows():
        temp_sum += entropy(list(row))
        count += 1
    mean_entroy = round(temp_sum/count,3)
    return mean_entroy

def defense_step_1():
    print(f"target_class: {target_label}")
    print(f"poisoned num: {len(purePoisonedTrainDataset)}")
    print(f"clean num: {len(pureTargetClassCleanTrainset)}")
    poisoned_mutated_models_acc, poisoned_mutated_models_predict_label_dic = get_mutated_model_predict_label(purePoisonedTrainDataset)
    print(f"poisoned_mutated_models_acc:{poisoned_mutated_models_acc}")
    df = pd.DataFrame(poisoned_mutated_models_predict_label_dic)
    save_file_name= "poisoned_mutated_models_predict_label_dic.csv"
    save_file_path = os.path.join(work_dir, save_file_name)
    df.to_csv(save_file_path, index=False)
    print(f"poisoned_mutated_models_predict_label_dic 保存在{save_file_path}")
    clean_mutated_models_acc, clean_mutated_models_predict_label_dic = get_mutated_model_predict_label(pureTargetClassCleanTrainset)
    print(f"clean_mutated_models_acc:{clean_mutated_models_acc}")
    df = pd.DataFrame(clean_mutated_models_predict_label_dic)
    save_file_name= "clean_mutated_models_predict_label_dic.csv"
    save_file_path = os.path.join(work_dir, save_file_name)
    df.to_csv(save_file_path, index=False)
    print(f"clean_mutated_models_predict_label_dic 保存在{save_file_path}")

def defense_step_2():
    poisoned_df = pd.read_csv(os.path.join(work_dir, "poisoned_mutated_models_predict_label_dic.csv"))
    clean_df = pd.read_csv(os.path.join(work_dir, "clean_mutated_models_predict_label_dic.csv"))
    poisoned_entropy =  caclue_entropy(poisoned_df)
    clean_entropy =  caclue_entropy(clean_df)
    print(f"poisoned_entropy:{poisoned_entropy}")
    print(f"clean_entropy:{clean_entropy}")


if __name__ == "__main__":

    target_label = identify_target_label()

    # defense_step_1()
    # defense_step_2()
    # detect(threshold=0.01)
    pass