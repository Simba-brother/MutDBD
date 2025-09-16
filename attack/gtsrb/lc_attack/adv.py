
from tqdm import tqdm
import random
from attack.gtsrb.lc_attack.pgd import PGD
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
from attack.gtsrb.lc_attack.custom_dataset import CustomImageDataset

def select_indices_to_adv(clean_trainset,target_class, percent):
    all_labels = []
    # 数据加载器
    batch_size = 128
    data_loader = DataLoader(
        clean_trainset,
        batch_size=batch_size,
        shuffle=False, # importent 
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    for batch_id, batch in enumerate(tqdm(data_loader,desc="数据集标签收集")):
        all_labels.extend(batch[1].tolist())
    # 选择出被投毒的sample_indices
    target_indices = []
    for i,label in enumerate(all_labels):
        if label == target_class:
            target_indices.append(i)
    select_num = max(1,int(len(target_indices) * percent))
    selected_indices = random.sample(target_indices, select_num)
    return selected_indices,target_indices

def split_dataset(clean_trainset, selected_indices):
    all_indices = set(range(len(clean_trainset)))
    indices_to_keep = list(all_indices-set(selected_indices))
    origin_subset = Subset(clean_trainset, indices_to_keep)
    to_adv_subset = Subset(clean_trainset, selected_indices)
    return origin_subset,to_adv_subset

def get_adv_dataset(victim_model, to_adv_dataset, device):
    data_loader = DataLoader(
        to_adv_dataset,
        batch_size=128,
        shuffle=False, # importent 
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    loss_fn = nn.CrossEntropyLoss()
    steps = 100
    alpha = 0.01
    epsilon = 0.3 # 0.3
    # 开始对抗攻击
    pgd = PGD(victim_model,loss_fn,steps,alpha,epsilon)
    success_num = 0 
    total = 0
    all_adv_X = []
    all_labels = []
    all_pred_labels = []
    for batch_id, batch in enumerate(tqdm(data_loader,desc="对抗攻击中")):
        X = batch[0].to(device)
        Y = batch[1].to(device)
        total += X.shape[0]
        adv_X = pgd.perturb(X,Y,device)
        with torch.no_grad():
            outputs = victim_model(adv_X)
            pred_Y = torch.argmax(outputs, dim=1)
            all_adv_X.append(adv_X)
            all_labels.append(Y)
            all_pred_labels.append(pred_Y)
            success_num += (pred_Y != Y).sum()
    combined_adv_X = torch.cat(all_adv_X,dim=0)
    combined_labels = torch.cat(all_labels,dim=0)
    combined_pred_labels = torch.cat(all_pred_labels,dim=0)
    mapping_list = (combined_labels == combined_pred_labels).tolist()
    # 获取所有 True 值的索引
    unsuccess_indices = set([i for i, value in enumerate(mapping_list) if value is True])
    adv_asr = round(success_num.item()/total,4)
    adv_dataset = CustomImageDataset(combined_adv_X,combined_labels)

    data_loader = DataLoader(
        adv_dataset,
        batch_size=1,
        shuffle=False, # importent 
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    return adv_dataset,adv_asr,unsuccess_indices


def construt_fusion_dataset(victim_model,clean_trainset,device,target_class,percent):
    # 选择将要对抗的样本
    selected_indices,target_indices = select_indices_to_adv(clean_trainset,target_class, percent)
    print(f"选择的数据量/target数据量:{len(selected_indices)}/{len(target_indices)}")
    print(f"选择的数据量/总数据量:{len(selected_indices)}/{len(clean_trainset)}")

    # 将数据切分为原始训练集部分和将要对抗部分
    origin_subset,to_adv_subset = split_dataset(clean_trainset, selected_indices)

    # 开始对抗
    adv_subset,adv_asr,unsuccess_indices = get_adv_dataset(victim_model, to_adv_subset, device)
    print(f"对抗成功率:{adv_asr},对抗失败数据量/对抗总量:{len(unsuccess_indices)/len(to_adv_subset)}")

    # 构建混和(干净+对抗)数据集
    fusion_dataset = ConcatDataset([origin_subset,adv_subset])
    adv_indices = []
    M = len(origin_subset)
    N = len(adv_subset)
    adv_indices = list(range(M, M + N))
    assert len(adv_indices) == len(adv_subset), "数据错误"
    return fusion_dataset,selected_indices,adv_indices

    


    
