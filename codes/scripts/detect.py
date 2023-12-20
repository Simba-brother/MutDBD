import sys
sys.path.append("./")
import random
import os
import statistics
import joblib
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader,Dataset
from codes.datasets.cifar10.models.resnet18_32_32_3 import ResNet
from codes.datasets.cifar10.models.vgg import VGG
from codes.eval_model import EvalModel
from codes.utils import entropy, create_dir

random.seed(555)
model = VGG("VGG19")
attack_name = "WaNet" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet
mutation_ratio = 0.05 # importent!!
exp_root_dir = "/data/mml/backdoor_detect/experiments"
dataset_name = "CIFAR10"
model_name = "vgg19" #resnet18_nopretrain_32_32_3, vgg19

mutates_path = os.path.join(exp_root_dir, dataset_name, model_name, "mutates")
mutation_name_list = ["gf","neuron_activation_inverse","neuron_block","neuron_switch","weight_shuffle"]


if dataset_name == "CIFAR10":
    if model_name == "resnet18_nopretrain_32_32_3":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import *
    if model_name == "vgg19":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_vgg19 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_vgg19 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_vgg19 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_vgg19 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_vgg19 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_vgg19 import *

dict_state = get_dict_state()
target_class_idx = 1

class TargetClassDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx  = target_class_idx
        self.target_class_dataset = self.get_target_class_dataset()

    def get_target_class_dataset(self):
        target_class_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label == self.target_class_idx:
                target_class_dataset.append((sample, label))
        return target_class_dataset
    
    def __len__(self):
        return len(self.target_class_dataset)
    
    def __getitem__(self, index):
        x,y=self.target_class_dataset[index]
        return x,y
    
class NoTargetClassDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx  = target_class_idx
        self.notarget_class_dataset = self.get_target_class_dataset()

    def get_target_class_dataset(self):
        notarget_class_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label != self.target_class_idx:
                notarget_class_dataset.append((sample, label))
        return notarget_class_dataset
    
    def __len__(self):
        return len(self.notarget_class_dataset)
    
    def __getitem__(self, index):
        x,y=self.notarget_class_dataset[index]
        return x,y
    
pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
poisoned_trainset = dict_state["poisoned_trainset"]
target_class_dataset_clean = TargetClassDataset(pureCleanTrainDataset, target_class_idx)
target_class_dataset_poisoned = TargetClassDataset(purePoisonedTrainDataset, target_class_idx)
no_target_class_trainset = NoTargetClassDataset(poisoned_trainset, target_class_idx)

device = torch.device("cuda:0")



def get_mutation_models_weight_file_path():
    weight_filePath_list = []
    for mutation_name in mutation_name_list:
        mutation_name_dir_path = os.path.join(mutates_path, mutation_name)
        if mutation_name  == "gf":
            ratio_dir = f"ratio_{mutation_ratio}_scale_5_num_50"
        else:
            ratio_dir = f"ratio_{mutation_ratio}_num_50"
        mutated_models_weight_dir_path = os.path.join(mutation_name_dir_path, ratio_dir, attack_name) 
        for m_i in range(50):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"model_mutated_{m_i+1}.pth")
            weight_filePath_list.append(weight_path)
    assert len(weight_filePath_list) == len(mutation_name_list) * 50, "出错了,总的变异数目应该是250"
    print("get_mutation_models_weight_file_path() successfully")
    return weight_filePath_list




def get_pred_label(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        pred_labels = e._get_pred_labels()
        res[f"m_{m_i}"] = pred_labels
    df = pd.DataFrame(res)
    print("get_pred_label() successfully")
    return df


def get_output(weight_filePath_list:list, dataset):
    res = {}
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        outputs = e._get_outputs()
        res[f"m_{m_i}"] = outputs
    print("get_output() successfully")
    return res

def get_mean_entropy(df:pd.DataFrame):
    entropy_list = []
    for row_id, row in df.iterrows():
        entropy_list.append(entropy(list(row)))
    average = statistics.mean(entropy_list)
    print("get_mean_entropy() successfully")
    return average

def sort_model_by_acc(weight_filePath_list:list, dataset):
    '''
    acc越低,模型优先级越高
    '''
    sorted_weight_filePath_list = []
    acc_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        acc = e._eval_acc()
        acc_list.append(acc)
    sorted_model_indices = sorted(range(len(acc_list)), key=lambda x: acc_list[x])
    for i in sorted_model_indices:
        sorted_weight_filePath_list.append(weight_filePath_list[i])
    return sorted_weight_filePath_list


def calu_deepGini(output):
    cur_sum = 0
    for v in output:
        cur_sum += v*v
    return 1-cur_sum

def sort_model_by_deepGini(weight_filePath_list:list, dataset):
    '''
    deepgini越大,优先级越高
    '''
    sorted_weight_filePath_list = []
    deepGini_list = []
    for m_i, weight_filePath in enumerate(weight_filePath_list):
        weight = torch.load(weight_filePath, map_location="cpu")
        model.load_state_dict(weight)
        e = EvalModel(model, dataset, device)
        outputs = e._get_outputs()
        avg_deepGini = 0
        for output in outputs:
            avg_deepGini += calu_deepGini(output)
        avg_deepGini = avg_deepGini / len(outputs)
        deepGini_list.append(avg_deepGini)
    sorted_model_indices = sorted(range(len(deepGini_list)), key=lambda x: deepGini_list[x])
    sorted_model_indices.reverse()
    for i in sorted_model_indices:
        sorted_weight_filePath_list.append(weight_filePath_list[i])
    return sorted_weight_filePath_list

def select_top_k_models(sorted_weight_filePath_list, k=50):
    cut_off = 0.5
    cut_point = int(len(sorted_weight_filePath_list)*cut_off)
    temp_list = sorted_weight_filePath_list[:cut_point]
    ans = random.sample(temp_list, k)
    return ans



def detect(threshold):
    clean_csv_path = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, "clean_trainset_pred_labels.csv")
    poisoned_csv_path = os.path.join(exp_root_dir, dataset_name, model_name, attack_name, "poisoned_trainset_pred_labels.csv")
    clean_df = pd.read_csv(clean_csv_path)
    poisoned_df = pd.read_csv(poisoned_csv_path)

    ground_truth_list = []
    pred_list = []

    for row_idx, row in clean_df.iterrows():
        sample_entropy  = entropy(list(row))
        if sample_entropy < threshold:
            pred_list.append(True) # Poisoned
        else:
            pred_list.append(False) # clean
        ground_truth_list.append(False)
    
    for row_idx, row in poisoned_df.iterrows():
        sample_entropy  = entropy(list(row))
        if sample_entropy < threshold:
            pred_list.append(True) # Poisoned
        else:
            pred_list.append(False) # clean
        ground_truth_list.append(False)
    TN, FP, FN, TP = confusion_matrix(ground_truth_list, pred_list).ravel()
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


def save_df(df, save_dir, save_file_name):
    create_dir(save_dir)
    save_path = os.path.join(save_dir, save_file_name)
    df.to_csv(save_path, index=False)

def look_entropy():
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path()
    # 得到预测标签
    df = get_pred_label(weight_filePath_list, target_class_dataset_poisoned)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_poisoned_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    poisoned_average_entropy = get_mean_entropy(df)
    print("poisoned_average_entropy:", poisoned_average_entropy)


    df = get_pred_label(weight_filePath_list, target_class_dataset_clean)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_clean_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    clean_average_entropy = get_mean_entropy(df)
    print("clean_average_entropy:", clean_average_entropy)

    print("==="*10)
    print("clean_average_entropy:", clean_average_entropy)
    print("poisoned_average_entropy:", poisoned_average_entropy)

def look_sorted_entropy():
    # 加载adaptive_ratio model weights
    weight_filePath_list = get_mutation_models_weight_file_path()
    # 根据mutated model在clean samples上的acc,来排序mutated model
    sorted_weight_filePath_list = sort_model_by_acc(weight_filePath_list, no_target_class_trainset)
    # 取top
    selected_weight_filePath_list = select_top_k_models(sorted_weight_filePath_list, k=50)
    # 得到预测标签
    df = get_pred_label(selected_weight_filePath_list, target_class_dataset_poisoned)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_poisoned_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    poisoned_average_entropy = get_mean_entropy(df)
    print("poisoned_average_entropy:", poisoned_average_entropy)


    df = get_pred_label(selected_weight_filePath_list, target_class_dataset_clean)
    # 保存预测结果
    save_dir = os.path.join(exp_root_dir, dataset_name, model_name, attack_name)
    save_file_name = "sorted_clean_trainset_pred_labels.csv"
    save_df(df, save_dir, save_file_name)
    # 计算平均熵
    clean_average_entropy = get_mean_entropy(df)
    print("clean_average_entropy:", clean_average_entropy)

    print("==="*10)
    print("clean_average_entropy:", clean_average_entropy)
    print("poisoned_average_entropy:", poisoned_average_entropy)

if __name__ == "__main__":
    # look_entropy()
    look_sorted_entropy()

