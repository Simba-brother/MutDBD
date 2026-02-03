import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from models.model_loader import get_model
from utils.common_utils import set_random_seed,get_formattedDateTime
from datasets.clean_dataset import get_clean_dataset
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
import time

def modify_bias(model,dataset_name,model_name,target_class, delta=5.0):
    """第二步：根据其他bias值智能调整target class的bias"""
    print(f"智能调整target class {target_class}的bias值...")

    # 找到最后一层
    if dataset_name == "CIFAR10" and model_name == "ResNet18":
        last_layer = model.classifier
    else:
        raise ValueError("无法找到模型的最后一层")

    # 获取所有bias
    if last_layer.bias is not None:
        with torch.no_grad():
            all_biases = last_layer.bias.clone()

            # 获取其他类别的bias（排除target class）
            other_biases = torch.cat([all_biases[:target_class], all_biases[target_class+1:]])
            max_other_bias = other_biases.max().item()

            # 计算需要增加的值：让target class的bias略高于其他bias的最大值
            current_target_bias = all_biases[target_class].item()
            needed_increase = 20 # max(0, max_other_bias - current_target_bias + delta)

            # 修改target class的bias
            last_layer.bias[target_class] += needed_increase

            print(f"其他bias最大值: {max_other_bias:.4f}")
            print(f"target class原始bias: {current_target_bias:.4f}")
            print(f"增加量: {needed_increase:.4f}")
            print(f"target class新bias: {last_layer.bias[target_class].item():.4f}")
    else:
        raise ValueError("最后一层没有bias参数")

    return model


def evaluate_model_acc_asr(model, testset, target_class, device, batch_size=128):
    """评估模型性能"""
    model.eval()
    model.to(device)

    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    gt_labels = []
    p_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            gt_labels.extend(labels.tolist())
            p_labels.extend(predicted.tolist())

    no_target_class_gt_labels = []
    no_target_class_p_labels = []
    for gt_label,p_label in zip(gt_labels,p_labels):
        if gt_label != target_class:
            no_target_class_gt_labels.append(gt_label)
            no_target_class_p_labels.append(p_label)

    correct = 0
    for g_label,p_label in zip(gt_labels,p_labels):
        if p_label == g_label:
            correct += 1
    clean_acc = round(correct / len(no_target_class_gt_labels),4)

    attack_success = 0
    for g_label,p_label in zip(no_target_class_gt_labels,no_target_class_p_labels):
        if p_label == target_class:
            attack_success += 1
    asr = round(attack_success/len(no_target_class_gt_labels),4)

    return clean_acc,asr


def attack(dataset_name,model_name,attack_name,save_dir):
    # 得到trained clean model
    model = get_model(dataset_name, model_name)
    # load trained clean state_dict
    state_dict = torch.load(os.path.join(exp_root_dir,"BenignTrain",dataset_name,model_name,"best_model_epoch_42.pth"),map_location="cpu")
    trained_clean_model_state_dict = state_dict["model_state_dict"]
    model.load_state_dict(trained_clean_model_state_dict)
    # 第二步：智能调整target class的bias
    backdoor_model = modify_bias(model,dataset_name,model_name, target_class, delta=1.0)
    alltrainset, testset = get_clean_dataset(dataset_name,attack_name)
    device = torch.device(f"cuda:{gpu_id}")

    clean_acc,asr = evaluate_model_acc_asr(model, testset, target_class, device, batch_size=128)


    # 保存后门模型和数据
    backdoor_data = {}
    backdoor_data["backdoor_model_weights"] = backdoor_model.state_dict()
    

    backdoor_data['target_class'] = target_class
    backdoor_data['clean_acc'] = clean_acc
    backdoor_data['asr'] = asr

    save_path = os.path.join(save_dir, "backdoor_data.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(backdoor_data, save_path)
    print(f"SBA攻击完成,数据被存入{save_path}")
    print(f"干净准确率: {clean_acc:.4f}, ASR: {asr:.4f}")


if __name__ == "__main__":
    
    exp_name = "ATTACK"
    exp_time = get_formattedDateTime()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_save_dir = os.path.join(exp_root_dir,exp_name)
    os.makedirs(exp_save_dir,exist_ok=True)

    print("exp_name:",exp_name)
    print("exp_time:",exp_time)
    print("exp_root_dir:",exp_root_dir)
    print("exp_save_dir:",exp_save_dir)


    r_seed = 42
    dataset_name = "CIFAR10"
    attack_name = "SBA"
    model_name = "ResNet18"
    target_class = 3
    gpu_id = 1

    print("r_seed:",r_seed)
    print("gpu_id:",gpu_id)
    
    print(f"{dataset_name}|{model_name}|{attack_name}")
    set_random_seed(r_seed)
    save_dir = os.path.join(exp_save_dir,dataset_name,model_name,attack_name)
    os.makedirs(save_dir,exist_ok=True)
    attack(dataset_name,model_name,attack_name,save_dir)
