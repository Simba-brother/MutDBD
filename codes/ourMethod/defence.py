'''
重要
我们基于ASD的defence方法
'''

import joblib
import random
import os
import pandas as pd
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # 用于批量加载训练集的

import matplotlib.pyplot as plt

from codes import config
from codes.asd.loss import SCELoss, MixMatchLoss
from codes.asd.semi import poison_linear_record, mixmatch_train,linear_test
from codes.asd.dataset import MixMatchDataset

from codes.utils import create_dir


def sampling_analyse(loss_array,poisoned_ids,sample_idx_array,gt_label_array):
    '''
    对采样数据进行分析和可视化
    Args
    ----------
    loss_array:ndarray
        每个样本在模型上的loss值。例如，loss_array[idx]即可获得样本idx的loss值。
    poisoned_ids:list
        中毒样本的idx。
    sample_idx_array:ndarray
        被采样的样本idx即准备到clean pool中的样本idx
    gt_label_array:ndarray
        每个样本的真实分类标签，例如，gt_label_array[idx]即可获得样本idx的真实分类标签。
    '''
    '''
    分析1：分析下按照loss值排序的中毒样本的累计增长趋势，看看符不符合中毒样本大概率集中在loss值大的样本
    '''
    # 根据loss值从小到大排序样本idx
    ranked_sample_idx_array =  loss_array.argsort()
    # 基于ranked_sample_idx_array和poisoned_ids构建对应的ispoisoned_list
    ispoisoned_list = []
    for sample_idx in ranked_sample_idx_array:
        if sample_idx in poisoned_ids:
            ispoisoned_list.append(1) # 1 代表污染
        else:
            ispoisoned_list.append(0)
    cumulative_count_list = []
    count = 0
    for loc in range(len(ispoisoned_list)):
        count += ispoisoned_list[loc]
        cumulative_count_list.append(count)
    # 计算累积百分比（相对于总样本数）
    total_samples = len(ispoisoned_list)
    cumulative_percent_list = [count / total_samples for count in cumulative_count_list]

    # 绘制曲线
    # 设置画布大小
    plt.figure(figsize=(10, 6))
    # 绘制累积数量曲线
    plt.plot(range(1, len(ispoisoned_list)+1), cumulative_percent_list, label='Cumulative number of poisoned data', marker='o')

    # 添加图表元素
    plt.xlabel('Sorting position (1st to {}th)'.format(len(ispoisoned_list)))
    plt.ylabel('Cumulative number')
    plt.title('Growth curve of poisoned data after sorting')
    plt.legend()
    plt.grid(True)
    plt.savefig("imgs/ASD_1.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()
    '''
    分析2：分析下采样出的样本排名分布，因为ASD是直接基于排名进行采样的，即直接采样排名前50%的样本，
    而我们是根据样本的gt_label为每个样本设置了采样概率，所以需要观察一下我们采样的样本的排名，因为
    根据分析1发现越靠后越可能为中毒样本。
    '''
    # 记录排名分布，1表示对应该排名位置的样本被采样了，反之不然。例如，如果rank_distribution[0]=1则说明排名第一的样本被采样了。
    rank_distribution = [0]*len(ranked_sample_idx_array)
    # 遍历每个位次
    for rank in range(len(ranked_sample_idx_array)):
        # 当前位次的样本索引（item）
        item = ranked_sample_idx_array[rank]
        # 判断当前位次样本在不在采样list中
        if item in sample_idx_array:
            # 如果该位次的样本被采样了
            rank_distribution[rank] = 1
    # 绘制热力图
    plt.imshow([rank_distribution], aspect='auto', cmap='Reds', interpolation='nearest')
    plt.title('Heat map distribution of poisoned samples')
    plt.xlabel('Position Index')
    plt.colorbar()
    plt.yticks([])
    plt.savefig("imgs/ASD_2.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()
    '''
    分析2：分析下采样出的样本的类别统计分布。因为，我们的方法考虑了样本的标签，不同的标签具有不同的采样概率，因此需要
    统计一下类别统计分布，看看target class的样本占比。
    '''
    # 用于存储被采样样本的gt_label
    label_list = []
    for sample_idx in sample_idx_array:
        label = gt_label_array[sample_idx]
        label_list.append(label)
    # 统计每个类别绝对数量
    label_counts = pd.Series(label_list).value_counts()  
    # 统计每个类别百分比
    label_percent = pd.Series(label_list).value_counts(normalize=True).sort_index()
    # ====== 绘制柱状图 ======
    plt.figure(figsize=(10, 6))
    sorted_labels = label_percent.index  # 确保标签顺序正确

    # 绘制柱状图并添加百分比标签
    bars = plt.bar(
        x=sorted_labels,
        height=label_percent.values,
        color='skyblue',
        edgecolor='black'
    )

    # 在柱顶显示百分比（保留1位小数）
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,  # 横坐标居中
            height + 0.02,  # 纵坐标偏移（根据比例调整）
            f'{height:.1%}',  # 显示百分比格式
            ha='center', va='bottom',  # 水平居中，底部对齐
            fontsize=9
        )

    plt.title('Label distribution (proportion)', fontsize=14)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xticks(ticks=sorted_labels)  # 显式指定坐标轴刻度
    plt.ylim(0, 1.2 * max(label_percent))  # 扩大Y轴范围避免文字溢出
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("imgs/ASD_3.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.close()

def get_class_sampled_prob_map(classes_rank:list):
    '''
    根据类别排序（即，从左往右类别的可疑程度逐渐减轻）得到类别对应的采样概率。
    '''
    classes_num = len(classes_rank)
    class_map = {}
    intervals = []
    for cut_rate in [0.25,0.5,0.75]:
        intervals.append(int(cut_rate*classes_num))
    for i in range(classes_num):
        cls = classes_rank[i]
        if i <= intervals[0]:
            # [0,25%]
            prob = 0.25
        elif i <= intervals[1]:
            # (25%,50%]
            prob = 0.5
        elif i <= intervals[2]:
            # (50%,75%]
            prob = 0.75
        else:
            # (75%,100%]
            prob = 1
        class_map[cls] = prob
    return class_map

def sampling(samples_num:int,ranked_sample_idx_array,label_prob_map:dict,label_list):
    '''
    采样
    Args:
        samples_num (int): 采样的数量
        ranked_sample_idx_array (1dArray):排序的样本id array
        label_prob_map (dict):样本标签到采样概率的映射
        label_list（1dArray）:样本标签array
    '''
    # 选择的是clean sample
    choice_indice = []
    while len(choice_indice) < samples_num:
        for sample_idx in ranked_sample_idx_array:
            # 得到该样本被采样概率
            prob = label_prob_map[label_list[sample_idx]]
            cur_p =  random.random()
            if cur_p < prob:
                # 概率出现,该样本进入total_indice
                choice_indice.append(sample_idx)
                if len(choice_indice) == samples_num:
                    # 如果数量够了直接break
                    break
    assert len(choice_indice) == samples_num, "数量不对"
    return choice_indice

def sampling_2(samples_num:int,ranked_sample_idx_array, cls_rank, label_list) -> list:
    # 排名0-49999
    loc_list = list(range(len(ranked_sample_idx_array)))
    # 位次对应的标签
    loc_label_list = []
    for loc in loc_list:
        # 该位次样本id
        sample_id = ranked_sample_idx_array[loc]
        # 该位次样本label
        loc_label_list.append(label_list[sample_id])
    
    def calculate_weight(idx, label, priority_order):
        """计算归一化权重"""
        max_index = 49999
        max_coeff = 10.0
        
        # 各类别系数分配（优先级越高的类别系数越大）
        coeff_dict = {}
        for rank, cat in enumerate(priority_order):
            coeff_dict[cat] = (10 - rank)/max_coeff  # 归一化到0-1
        
        # 索引归一化到0-1
        index_norm = idx / max_index
        
        # 权重 = 系数 × 索引归一化值
        weight = coeff_dict[label] * index_norm
        return weight
    def sample_low_risk(list1, labels, priority_order):
        """按权重排序采样"""
        samples = []
        for idx, label in zip(list1, labels):
            weight = calculate_weight(idx, label, priority_order)
            samples.append( (weight, idx) )
        
        # 按权重从小到大排序，取前25000
        samples.sort(key=lambda x: x[0])
        selected_indices = [s[1] for s in samples[:25000]]
        
        return selected_indices
    selected_loc_list = sample_low_risk(loc_list, loc_label_list, cls_rank)

    seletcted_sample_id_list = []
    for loc in selected_loc_list:
       s_id = ranked_sample_idx_array[loc]
       seletcted_sample_id_list.append(s_id)
    return seletcted_sample_id_list


def defence_train(
        model, # victim model
        class_num, # 分类数量
        poisoned_train_dataset, # 有污染的训练集,不打乱的list
        poisoned_ids, # 被污染的样本id list
        poisoned_eval_dataset_loader, # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader, #有污染的训练集加载器,打乱顺序加载
        clean_test_dataset_loader, # 干净的测试集加载器
        poisoned_test_dataset_loader, # 污染的测试集加载器
        device, # GPU设备对象
        save_dir, # 实验结果存储目录 save_dir = os.path.join(exp_root_dir, "ASD", dataset_name, model_name, attack_name)
        
        **kwargs
        ):
    '''
    OurMethod防御训练方法
    '''
    # 类别排序，越靠前类别越有可能为target class
    classes_rank = kwargs["classes_rank"]
    #  各个类别采样概率
    class_prob_map = get_class_sampled_prob_map(classes_rank)
    
    model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 损失函数对象放到gpu上
    criterion.to(device)
    # 用于分割的损失函数
    split_criterion = SCELoss(alpha=0.1, beta=1, num_classes=class_num)
    # 分割损失函数对象放到gpu上
    split_criterion.to(device)
    # semi 损失函数
    semi_criterion = MixMatchLoss(rampup_length=config.asd_config[kwargs["dataset_name"]]["epoch"], lambda_u=15) # rampup_length = 120  same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    # 模型参数的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    # 先选择clean seed
    # clean seed samples
    clean_data_info = {}
    all_data_info = {}
    for class_idx in range(class_num):
        clean_data_info[class_idx] = []
        all_data_info[class_idx] = []
    for idx, item in enumerate(poisoned_train_dataset):
        sample = item[0]
        label = item[1]
        if idx not in poisoned_ids:
            clean_data_info[label].append(idx)
        all_data_info[label].append(idx)
    # 选出的clean seed idx
    choice_clean_indice = []
    for class_idx, idx_list in clean_data_info.items():
        # 从每个class_idx中选择10个sample idx,replace表示无放回抽样
        choice_list = np.random.choice(idx_list, replace=False, size=10).tolist()
        choice_clean_indice.extend(choice_list)
        # 从all_data_info中剔除选择出的clean seed sample index
        all_data_info[class_idx] = [x for x in all_data_info[class_idx] if x not in choice_list]
    choice_clean_indice = np.array(choice_clean_indice)

    choice_num = 0
    best_acc = -1
    best_epoch = -1
    # 总共的训练轮次
    total_epoch = config.asd_config[kwargs["dataset_name"]]["epoch"]
    for epoch in range(total_epoch): # range(total_epoch): # range(60,90)
        print("===Epoch: {}/{}===".format(epoch+1, total_epoch))
        if epoch < 60: # epoch:[0,59]
            # 记录下样本的loss,feature,label,方便进行clean数据的挖掘
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device, dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"] )
            if epoch % 5 == 0 and epoch != 0:
                # 每五个epoch 每个class中选择数量就多加10个
                choice_num += 10
            print("Mining clean data by class-aware loss-guided split...")
            # all_data_info = {class_id:indice（剔除了干净种子）}
            # 0表示在污染池,1表示在clean pool
            split_indice = class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_ids)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < 90: # epoch:[60,89]
            # 使用此时训练状态的model对数据集进行record(记录下样本的loss,feature,label,方便进行clean数据的挖掘)
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device,dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])
            print("Mining clean data by class-agnostic loss-guided split...")
            # 将trainset对半划分为clean pool和poisoned pool
            split_indice =  class_agnostic_loss_guided_split(record_list, 0.5, poisoned_ids, class_prob_map=None, classes_rank = classes_rank) # class_prob_map=class_prob_map
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < total_epoch: # epoch:[90,120]
            # 使用此时训练状态的model对数据集进行record(记录下样本的loss,feature,label,方便进行clean数据的挖掘)
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device,dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])
            meta_virtual_model = deepcopy(model)
            dataset_name = kwargs["dataset_name"]
            model_name = kwargs["model_name"]
            if dataset_name in ["CIFAR10","GTSRB"]:
                if model_name == "ResNet18":
                    # 元虚拟模型要更新的参数
                    param_meta = [  
                                    {'params': meta_virtual_model.layer3.parameters()},
                                    {'params': meta_virtual_model.layer4.parameters()},
                                    # {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
                elif model_name == "VGG19":
                    param_meta = [  
                                    {'params': meta_virtual_model.classifier_1.parameters()},
                                    {'params': meta_virtual_model.classifier_2.parameters()},
                                ]
                elif model_name == "DenseNet":
                    param_meta = [  
                                    {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.dense4.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
            elif dataset_name == "ImageNet2012_subset":
                if model_name == "ResNet18":
                    # 元虚拟模型要更新的参数
                    param_meta = [  
                                    {'params': meta_virtual_model.layer3.parameters()},
                                    {'params': meta_virtual_model.layer4.parameters()},
                                    # {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.fc.parameters()}
                                ]
                elif model_name == "VGG19":
                    param_meta = [  
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
                elif model_name == "DenseNet":
                    param_meta = [  
                                    {'params': meta_virtual_model.features.denseblock4.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
            # 元模型的参数优化器
            meta_optimizer = torch.optim.Adam(param_meta, lr=0.015)
            # 元模型的损失函数
            meta_criterion = nn.CrossEntropyLoss(reduction="mean")
            meta_criterion.to(device)
            for _ in range(1):
                # 使用完整的训练集训练一轮元模型
                train_the_virtual_model(
                                        meta_virtual_model=meta_virtual_model, 
                                        poison_train_loader=poisoned_train_dataset_loader, 
                                        meta_optimizer=meta_optimizer,
                                        meta_criterion=meta_criterion,
                                        device = device
                                        )
            # 使用元模型对数据集进行一下record      
            meta_record_list = poison_linear_record(meta_virtual_model, poisoned_eval_dataset_loader, split_criterion, device, dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])

            # 开始干净样本的挖掘
            print("Mining clean data by meta-split...")
            split_indice = meta_split(record_list, meta_record_list, 0.5, poisoned_ids, class_prob_map=None, classes_rank = classes_rank)

            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)  

        # 开始clean pool进行监督学习,poisoned pool进行半监督学习    
        xloader = DataLoader(xdata,batch_size=64, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        uloader = DataLoader(udata,batch_size=64, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        print("MixMatch training...")
        # 半监督训练参数
        semi_mixmatch = {"train_iteration": 1024,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num}
        poison_train_result = mixmatch_train(
            model,
            xloader,
            uloader,
            semi_criterion,
            optimizer,
            epoch,
            device,
            **semi_mixmatch
        )

        print("Test model on clean data...")
        clean_test_result = linear_test(
            model, clean_test_dataset_loader, criterion,device
        )

        print("Test model on poison data...")
        poison_test_result = linear_test(
            model, poisoned_test_dataset_loader, criterion,device
        )

        # if scheduler is not None:
        #     scheduler.step()
        #     logger.info(
        #         "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
        #     )

        # Save result and checkpoint.
        # 保存结果
        result = {
            "poison_train": poison_train_result,
            "clean_test": clean_test_result,
            "poison_test": poison_test_result,
        }
        
        result_epochs_dir = os.path.join(save_dir, "result_epochs")
        create_dir(result_epochs_dir)
        save_file_name = f"result_epoch_{epoch}.data"
        save_file_path = os.path.join(result_epochs_dir, save_file_name)
        joblib.dump(result,save_file_path)
        print(f"epoch:{epoch},result: is saved in {save_file_path}")
        # result2csv(result, save_dir)
       
        # if scheduler is not None:
        #     saved_dict["scheduler_state_dict"] = scheduler.state_dict()

        is_best = False
        if clean_test_result["acc"] > best_acc:
            # 干净集测试acc的best
            is_best = True
            best_acc = clean_test_result["acc"]
            best_epoch = epoch
         # 保存状态
         # 每个训练轮次的状态
        saved_dict = {
            "epoch": epoch,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc, # clean testset上的acc
            "asr":poison_test_result["acc"],
            "best_epoch": best_epoch,
        }
        print("Best test accuaracy {} in epoch {}".format(best_acc, best_epoch))
        # 每当best acc更新后，保存checkpoint
        ckpt_dir = os.path.join(save_dir, "ckpt")
        create_dir(ckpt_dir)
        if is_best:
            # clean testset acc的best model
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(saved_dict, best_ckpt_path)
            print("Save the best model to {}".format(best_ckpt_path))
        # 保存最新一次checkpoint
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_model.pt")
        torch.save(saved_dict, latest_ckpt_path)
        print("Save the latest model to {}".format(latest_ckpt_path))

        # 保存第59个epoch(eg.stage1（class aware训练结束点）)
        if epoch == 59:
            latest_ckpt_path = os.path.join(ckpt_dir, "epoch59.pt")
            torch.save(saved_dict, latest_ckpt_path)
            print("Save the latest model to {}".format(latest_ckpt_path))
        # 保存第60个epoch(eg.stage1（class agnostic首个保存点）)
        if epoch == 60:
            latest_ckpt_path = os.path.join(ckpt_dir, "epoch60.pt")
            torch.save(saved_dict, latest_ckpt_path)
            print("Save the latest model to {}".format(latest_ckpt_path))
            
    print("OurMethod_train() End")
    return best_ckpt_path,latest_ckpt_path

def class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_indice):
    """
    Adaptively split the poisoned dataset by class-aware loss-guided split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    clean_pool_flag = np.zeros(len(loss)) # 每次选择都清空
    # 存总共选择的clean 的idx,包括seed和loss最低的的sample idx
    total_indice = choice_clean_indice.tolist() # choice_clean_indice装的seed
    for class_idx, sample_indice in all_data_info.items():
        # 这里的sample_indice是剔除了干净seed
        # 遍历每个class_idx
        sample_indice = np.array(sample_indice)
        loss_class = loss[sample_indice]
        # 选择SCE loss较低的
        indice_class = loss_class.argsort()[: choice_num]
        indice = sample_indice[indice_class]
        # list的extend操作
        total_indice += indice.tolist()
    # 统计构建出的clean pool 其中可能还混有污染样本的数量,这里我们统计一下
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count+=1
    total_indice = np.array(total_indice)
    clean_pool_flag[total_indice] = 1 # 1表示clean

    print(
        "{}/{} poisoned samples in clean data pool".format(poisoned_count, clean_pool_flag.sum())
    )
    return clean_pool_flag

'''
# 按照loss值对样本idx进行排序
ranked_sample_idx_array =  loss.argsort()
# 采样准备放入clean pool的
samples_num = int(len(ranked_sample_idx_array)*rate)
choice_idx_list = sampling(samples_num,ranked_sample_idx_array,class_prob_map,gt_label_array)
'''
def class_agnostic_loss_guided_split(record_list, ratio, poisoned_indice, class_prob_map=None, classes_rank=None):
    """
    Adaptively split the poisoned dataset by class-agnostic loss-guided split.

    Args:
    ----
    class_prob_map和classes_rank不同同时存在，因为代表了不同的采样
    """
    keys = [record.name for record in record_list]
    # 样本对应的loss值
    loss = record_list[keys.index("loss")].data.numpy()
    # 得到样本对应的ground truth label
    gt_label_array = record_list[keys.index("target")].data.numpy()
    # 申请出一个flag array，1为干净，0为污
    clean_pool_flag = np.zeros(len(loss))
    if class_prob_map is not None:
        # 按照loss值对样本idx进行排序，loss保持不变
        ranked_sample_idx_array =  loss.argsort()
        # 计算采样数
        samples_num = int(len(ranked_sample_idx_array)*ratio)
        total_indice = sampling(samples_num,ranked_sample_idx_array,class_prob_map,gt_label_array)
    elif classes_rank is not None:
        # 按照loss值对样本idx进行排序，loss保持不变
        ranked_sample_idx_array =  loss.argsort()
        # 计算采样数
        samples_num = int(len(ranked_sample_idx_array)*ratio)
        total_indice = sampling_2(samples_num,ranked_sample_idx_array,classes_rank,gt_label_array)
    else:
        total_indice = loss.argsort()[: int(len(loss) * ratio)]
    # 统计构建出的clean pool 中还混有污染样本的数量
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count+=1
    print(
        "{}/{} poisoned samples in clean data pool".format(poisoned_count, len(total_indice))
    )
    clean_pool_flag[total_indice] = 1
    '''
    # 额外分析与可视化
    sampling_analyse(
        loss_array = loss,
        poisoned_ids = poisoned_indice,
        sample_idx_array = total_indice,
        gt_label_array = gt_label_array
        )
    '''
    return clean_pool_flag

def meta_split(record_list, meta_record_list, ratio, poisoned_indice, class_prob_map=None, classes_rank=None):
    """
    Adaptively split the poisoned dataset by meta-split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    meta_loss = meta_record_list[keys.index("loss")].data.numpy()
    # 得到样本对应的ground truth label
    gt_label_array = record_list[keys.index("target")].data.numpy()
    # 申请出一个池子，1为干净，0为污
    clean_pool_flag = np.zeros(len(loss))
    loss = loss - meta_loss
    # dif小的样本被选择为clean样本
    if class_prob_map is not None:
        # 按照loss值对样本idx进行排序
        ranked_sample_idx_array =  loss.argsort()
        # 计算采样数
        samples_num = int(len(ranked_sample_idx_array)*ratio)
        total_indice = sampling(samples_num,ranked_sample_idx_array,class_prob_map,gt_label_array)
    elif classes_rank is not None:
        # 按照loss值对样本idx进行排序，loss保持不变
        ranked_sample_idx_array =  loss.argsort()
        # 计算采样数
        samples_num = int(len(ranked_sample_idx_array)*ratio)
        total_indice = sampling_2(samples_num,ranked_sample_idx_array,classes_rank,gt_label_array)
    else:
        total_indice = loss.argsort()[: int(len(loss) * ratio)]
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count += 1
    print("{}/{} poisoned samples in clean data pool".format(poisoned_count, len(total_indice)))
    clean_pool_flag[total_indice] = 1
    predict_p_idx_list = loss.argsort()[int(len(loss) * ratio):]
    tp_num= len(set(predict_p_idx_list) & set(poisoned_indice))
    recall = round(tp_num/len(poisoned_indice),4)
    precision = round(tp_num / len(predict_p_idx_list),4)
    f1 = 2*recall*precision/(precision+recall+1e-10)
    print(f"recall:{recall},precison:{precision},f1:{f1}")
    return clean_pool_flag

def train_the_virtual_model(meta_virtual_model, poison_train_loader, meta_optimizer, meta_criterion, device):
    """
    Train the virtual model in meta-split.
    """
    meta_virtual_model.train()
    for batch_idx, batch in enumerate(poison_train_loader):
        data = batch[0]
        target = batch[1]
        data = data.to(device)
        target = target.to(device)
        # 优化器中的参数梯度清零
        meta_optimizer.zero_grad()
        output = meta_virtual_model(data)
        meta_criterion.reduction = "mean"
        # 损失函数
        loss = meta_criterion(output, target)
        # 损失函数对虚拟模型参数求导
        loss.backward()
        # 优化器中的参数更新
        meta_optimizer.step()
