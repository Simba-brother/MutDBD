# ASD baseline mian file

import joblib
import setproctitle
import random
import os
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader # 用于批量加载训练集的

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop
from torchvision.datasets import DatasetFolder
from codes import config
from codes.core.attacks import BadNets
from codes.core.models.resnet import ResNet
from codes.asd.loss import SCELoss, MixMatchLoss
from codes.asd.semi import poison_linear_record, mixmatch_train,linear_test
from codes.asd.dataset import MixMatchDataset

from codes.utils import create_dir
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset


def sampling(samples_num:int,ranked_sample_idx_array,label_prob_map:dict,label_list):
    '''
    采样
    Args:
        samples_num (int): 采样的数量
        ranked_sample_idx_array (1dArray):排序的样本id array
        label_prob_map (dict):样本标签到采样概率的映射
        label_list（1dArray）:样本标签array
    '''
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
    #  各个类别采样概率
    class_prob_map = kwargs["class_prob_map"]
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
        # 从每个class_idx中选择10个sample idx
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
    for epoch in range(total_epoch):
        print("===Epoch: {}/{}===".format(epoch+1, total_epoch))
        if epoch < 60:
            # 记录下样本的loss,feature,label,方便进行clean数据的挖掘
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device, dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"] )
            if epoch % 5 == 0 and epoch != 0:
                # 每五个epoch 每个class中选择数量就多加10个
                choice_num += 10
            print("Mining clean data by class-aware loss-guided split...")
            # 0表示在污染池,1表示在clean pool
            split_indice = class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_ids)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < 90:
            # 使用此时训练状态的model对数据集进行record(记录下样本的loss,feature,label,方便进行clean数据的挖掘)
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device,dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])
            print("Mining clean data by class-agnostic loss-guided split...")
            split_indice = class_agnostic_loss_guided_split(record_list, 0.5, poisoned_ids, class_prob_map=class_prob_map)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < total_epoch:
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
            split_indice = meta_split(record_list, meta_record_list, 0.5, poisoned_ids, class_prob_map=class_prob_map)

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
    
    print("asd_train() End")
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

def class_agnostic_loss_guided_split(record_list, ratio, poisoned_indice, class_prob_map=None):
    """
    Adaptively split the poisoned dataset by class-agnostic loss-guided split.
    """
    keys = [record.name for record in record_list]
    # 样本对应的loss值
    loss = record_list[keys.index("loss")].data.numpy()
    # 得到样本对应的ground truth label
    gt_label_array = record_list[keys.index("target")].data.numpy()
    # 申请出一个池子，1为干净，0为污
    clean_pool_flag = np.zeros(len(loss))
    if class_prob_map is not None:
        # 按照loss值对样本idx进行排序
        ranked_sample_idx_array =  loss.argsort()
        # 计算采样数
        samples_num = int(len(ranked_sample_idx_array)*ratio)
        total_indice = sampling(samples_num,ranked_sample_idx_array,class_prob_map,gt_label_array)
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
    return clean_pool_flag

def meta_split(record_list, meta_record_list, ratio, poisoned_indice, class_prob_map=None):
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
