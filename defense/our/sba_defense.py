import os
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


import numpy as np

import torch
from torch.utils.data import DataLoader,Subset
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn

from utils.common_utils import set_random_seed
from models.model_loader import get_model
from datasets.posisoned_dataset import get_clean_dataset

def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          lr_scheduler=None, class_weight = None, weight_decay=None, early_stop=False):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    if lr_scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,T_max=num_epoch,eta_min=1e-6)
    if class_weight is None:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss(class_weight.to(device))
    loss_function.to(device)
    optimal_loss = float('inf')
    best_model = model
    patience = 5
    count = 0
    for epoch in range(num_epoch):
        step_loss_list = []
        for _, batch in enumerate(dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
            step_loss_list.append(loss.item())
        if lr_scheduler:
            scheduler.step()
        epoch_loss = sum(step_loss_list) / len(step_loss_list)
        print(f"epoch:{epoch},loss:{epoch_loss}")
        if epoch_loss < optimal_loss:
            count = 0
            optimal_loss = epoch_loss
            best_model = copy.deepcopy(model)
        else:
            count += 1
            if early_stop and count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return model,best_model

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

def get_backdoor_data():
    # 获得backdoor_data
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name,attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    return backdoor_model


def select_seed(dataset):
    '''
    选择干净种子
    '''
    # 数据加载器
    dataset_loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    
    seed_dict = defaultdict(list)
    label_list = []
    for _, batch in enumerate(dataset_loader):
        Y = batch[1]
        label_list.extend(Y.tolist())

    for sample_id in range(len(dataset)):
        label = label_list[sample_id]
        seed_dict[label].append(sample_id)

    # 获得种子数据集
    seed_id_list = []
    for class_id,sample_id_list in seed_dict.items():
        seed_id_list.extend(np.random.choice(sample_id_list, replace=False, size=10).tolist())
    seedSet = Subset(dataset,seed_id_list)
    return seedSet,seed_id_list


def defense(backdoor_model, trainset):
    seedSet,seed_id_list = select_seed(trainset)
    last_defense_model, best_defense_model = train(
                backdoor_model,device, seedSet,
                num_epoch=30, lr=1e-3, batch_size=64)
    return best_defense_model


def main():
    # 加载后门模型
    backdoor_model = get_backdoor_data()
    # 加载数据集
    trainset, testset = get_clean_dataset(dataset_name,attack_name)
    # 评估backdoor model Acc and Asr
    acc,asr = evaluate_model_acc_asr(backdoor_model, testset, target_class, device)
    print("SBA backdoor model")
    print(f"\tASR:{asr},ACC:{acc}")

    # 开始防御
    defense_model = defense(backdoor_model, trainset)
    # 评估backdoor model Acc and Asr
    acc,asr = evaluate_model_acc_asr(defense_model, testset, target_class, device)
    print("Defense model")
    print(f"\tASR:{asr},ACC:{acc}")

def vis(save_path):
    

    data = {
        "backdoor_model":{"ASR":0.59,"ACC":0.52},
        "defense_model":{"ASR":0.05,"ACC":0.93}
    }

    metrics = ["ASR", "ACC"]
    models = list(data.keys())

    values = np.array([[data[m][k] for k in metrics] for m in models])  # shape (2,2)

    x = np.array([0, 0.25])  # 进一步缩小两组柱子之间的间隔
    width = 0.05  # 让柱子更瘦

    # Colors: red for backdoor_model, green for defense_model
    colors = ["#ff2b2b", "#2f8f2f"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=800)

    bars1 = ax.bar(x - width/2, values[0], width, color=colors[0],
                   edgecolor="black", linewidth=1.6, label="backdoor model")
    bars2 = ax.bar(x + width/2, values[1], width, color=colors[1],
                   edgecolor="black", linewidth=1.6, label="defense model")

    # Axes labels
    ax.set_xlabel("Metrics", fontsize=28)
    ax.set_ylabel("Percentage (%)", fontsize=28)

    # Limits and ticks
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.1, 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    # Grid (horizontal dotted lines)
    ax.grid(axis="y", linestyle=":", linewidth=1.3, color="0.7")

    # Minor ticks and tick style (ticks on all sides, inward)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=9, width=2, labelsize=22, pad=10)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=5, width=1.5)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Value labels on top of bars
    for bars in [bars1, bars2]:
        for b in bars:
            height = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, height + 0.01, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=18)

    # Legend
    ax.legend(fontsize=20, frameon=True, edgecolor="black", fancybox=False)

    plt.tight_layout()
    
    plt.savefig(save_path,dpi=800)




if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "SBA"
    gpu_id = 1
    r_seed = 43
    device = torch.device(f"cuda:{gpu_id}")
    target_class = 3
    set_random_seed(r_seed)
    # main()
    save_path = "imgs/SBA/defense.pdf"
    vis(save_path)