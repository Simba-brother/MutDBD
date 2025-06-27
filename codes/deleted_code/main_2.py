# ASD baseline mian file
import sys
sys.path.append("./")
import joblib
import setproctitle
import os
import random
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # 用于批量加载训练集的

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop
from torchvision.datasets import DatasetFolder
from codes import config
from core.attacks import BadNets
from core.models.resnet import ResNet
from asd.loss import SCELoss, MixMatchLoss
from asd.semi import poison_linear_record, mixmatch_train,linear_test
from asd.dataset import MixMatchDataset
from asd.log import result2csv
from bigUtils import create_dir
from asd.models.resnet_cifar import get_model
from bigUtils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from dataset import PoisonLabelDataset, MixMatchDataset
def main():
    exp_root_dir = config.exp_root_dir
    dataset_name = config.dataset_name
    model_name = config.model_name
    attack_name = config.attack_name
    class_num = config.class_num
    proctitle = f"ASD|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    print(f"proctitle:{proctitle}")
    bd_config = {}
    bd_config["badnets"] = {}
    bd_config["badnets"]["trigger_path"] = "codes/core/attacks/BadNets_trigger.png"
    # 生成backdoor transform对象
    bd_transform = get_bd_transform(bd_config)
    target_label = 1
    poison_ratio = 0.1
    pre_transform = Compose([])
    train_primary_transform = Compose([
        RandomCrop(size=32,padding=4,padding_mode="reflect"),
        RandomHorizontalFlip(p=0.5)
    ])
    train_remaining_transform =  Compose([
        ToTensor()
    ])
    # 训练集图像处理
    train_transform = {
        "pre": pre_transform, # 预处理
        "primary": train_primary_transform, # 主处理
        "remaining": train_remaining_transform, # 剩余处理
    }
    test_primary_transform = Compose([])
    test_remaining_transform =  Compose([
        ToTensor()
    ])
     # 测试集图像处理
    test_transform = {
        "pre": pre_transform, # 预处理
        "primary": test_primary_transform, # 主处理
        "remaining": test_remaining_transform, # 剩余处理
    }
    # 获得干净的训练数据
    clean_train_data = get_dataset(
        "/data/mml/dataset/cifar-10-batches-py",
        train_transform, prefetch=False
    )
    # 获得干净的测试数据
    clean_test_data = get_dataset(
        "/data/mml/dataset/cifar-10-batches-py",
        test_transform, train=False, prefetch=False
    )
    # 先生成要污染的训练集实例id_list
    poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
    # 获得污染训练集
    poison_train_data = PoisonLabelDataset(
        clean_train_data, bd_transform, poison_train_idx, target_label)
    # 先生成要污染的测试集集实例id_list
    poison_test_idx = gen_poison_idx(clean_test_data, target_label)
    # 获得污染测试集
    poison_test_data = PoisonLabelDataset(
        clean_test_data, bd_transform, poison_test_idx, target_label
    )
    loader_params =  {
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True
    }
    # 污染训练集加载器
    poison_train_loader = get_loader(poison_train_data, loader_params, shuffle=True)
    # 使用训练集作为污染评估集
    poison_eval_loader = get_loader(poison_train_data, loader_params)
    # 干净测试集加载器
    clean_test_loader = get_loader(clean_test_data, loader_params)
    # 污染测试集
    poison_test_loader = get_loader(poison_test_data, loader_params)
    # 获得victim model
    model = get_model(class_num=10)
    # 获得GPU设备
    device = torch.device("cuda:0")
    # 将model放置设备上
    model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss(reduction = "mean")
    # 损失函数对象放到GPU上
    criterion.to(device)
    # split 损失函数
    # 用于分割的损失函数
    split_criterion = SCELoss(alpha=0.1, beta=1, num_classes=class_num)
    # 分割损失函数对象放到gpu上
    split_criterion.to(device)
    # semi 损失函数
    semi_criterion = MixMatchLoss(rampup_length=120, lambda_u=15) # rampup_length = 120 same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    # clean seed samples
    clean_data_info = {}
    all_data_info = {}
    for i in range(class_num): # 10
        clean_data_info[str(i)] = []
        all_data_info[str(i)] = []
    for idx, item in enumerate(poison_train_data):
        if item['poison'] == 0:
            clean_data_info[str(item['target'])].append(idx)
        all_data_info[str(item['target'])].append(idx)
    indice = []
    for k, v in clean_data_info.items():
        choice_list = np.random.choice(v, replace=False, size=10).tolist()
        indice = indice + choice_list
        # 剔除
        all_data_info[k] = [x for x in all_data_info[k] if x not in choice_list]
    # 存储了选择出的clean seed
    indice = np.array(indice)

    choice_num = 0
    best_acc = -1
    best_epoch = -1
    for epoch in range(120):
        print("===Epoch: {}/{}===".format(epoch, 120))
        if epoch < 60:
            record_list = poison_linear_record(model, poison_eval_loader, split_criterion, device)
            if epoch % 5 == 0 and epoch != 0:
                # 每五个epoch 就选择10个
                choice_num += 10
            print("Mining clean data by class-aware loss-guided split...")
            # 0表示在污染池,1表示在clean pool
            split_indice = class_aware_loss_guided_split(record_list, indice, all_data_info, choice_num, poison_train_idx)
            xdata = MixMatchDataset(poison_train_data, split_indice, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_indice, labeled=False)
        elif epoch < 90:       
            record_list = poison_linear_record(model, poison_eval_loader, split_criterion, device)
            print("Mining clean data by class-agnostic loss-guided split...")
            split_indice = class_agnostic_loss_guided_split(record_list, 0.5, poison_train_idx)
            xdata = MixMatchDataset(poison_train_data, split_indice, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_indice, labeled=False)
        elif epoch < 120:
            record_list = poison_linear_record(model, poison_eval_loader, split_criterion, device)
            meta_virtual_model = deepcopy(model)
            # param_meta = [
            #                 {'params': meta_virtual_model.linear.parameters()},
            #                 {'params': meta_virtual_model.classifier.parameters()}
            #             ]
            param_meta = [
                            {'params': meta_virtual_model.backbone.layer3.parameters()},
                            {'params': meta_virtual_model.backbone.layer4.parameters()},
                            {'params': meta_virtual_model.linear.parameters()}
                        ]
            meta_optimizer = torch.optim.Adam(param_meta, lr=0.015)
            meta_criterion = nn.CrossEntropyLoss(reduction = "mean")
            meta_criterion.to(device)
            for _ in range(1):
                train_the_virtual_model(
                                        meta_virtual_model=meta_virtual_model, 
                                        poison_train_loader=poison_train_loader, 
                                        meta_optimizer=meta_optimizer,
                                        meta_criterion=meta_criterion,
                                        device = device
                                        )      
            meta_record_list = poison_linear_record(meta_virtual_model, poison_eval_loader, split_criterion, device)

            print("Mining clean data by meta-split...")
            split_indice = meta_split(record_list, meta_record_list, 0.5, poison_train_idx)

            xdata = MixMatchDataset(poison_train_data, split_indice, labeled=True)
            udata = MixMatchDataset(poison_train_data, split_indice, labeled=False)  

        # 开始clean pool进行监督学习,poisoned pool进行半监督学习    
        xloader = DataLoader(xdata,batch_size=64, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        uloader = DataLoader(udata,batch_size=64, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        print("MixMatch training...")
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
            model, clean_test_loader, criterion,device
        )

        print("Test model on poison data...")
        poison_test_result = linear_test(
            model, poison_test_loader, criterion,device
        )

        # if scheduler is not None:
        #     scheduler.step()
        #     logger.info(
        #         "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
        #     )

        # Save result and checkpoint.
        
        result = {
            "poison_train": poison_train_result,
            "clean_test": clean_test_result,
            "poison_test": poison_test_result,
        }
        
        save_dir = os.path.join(exp_root_dir, "ASD", dataset_name, model_name, attack_name)
        create_dir(save_dir)
        save_file_name = f"result_epoch_{epoch}.data"
        save_file_path = os.path.join(save_dir, save_file_name)
        joblib.dump(result,save_file_path)
        print(f"epoch:{epoch},result: is saved in {save_file_path}")
        # result2csv(result, save_dir)

        saved_dict = {
            "epoch": epoch,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
        }
        # if scheduler is not None:
        #     saved_dict["scheduler_state_dict"] = scheduler.state_dict()

        is_best = False
        if clean_test_result["acc"] > best_acc:
            is_best = True
            best_acc = clean_test_result["acc"]
            best_epoch = epoch
        print("Best test accuaracy {} in epoch {}".format(best_acc, best_epoch))
        ckpt_dir = os.path.join(exp_root_dir, "ASD", dataset_name, model_name, attack_name, "ckpt")
        create_dir(ckpt_dir)
        if is_best:
            ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(saved_dict, ckpt_path)
            print("Save the best model to {}".format(ckpt_path))
        ckpt_path = os.path.join(ckpt_dir, "latest_model.pt")
        torch.save(saved_dict, ckpt_path)
        print("Save the latest model to {}".format(ckpt_path))
    print("main() End")

def class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_indice):
    """
    Adaptively split the poisoned dataset by class-aware loss-guided split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    # poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_flag = np.zeros(len(loss))
    # 存总共选择的clean 的idx,包括seed和loss最底的sample idx
    total_indice = choice_clean_indice.tolist()
    for class_idx, sample_indice in all_data_info.items():
        # 遍历每个class_idx
        sample_indice = np.array(sample_indice)
        loss_class = loss[sample_indice]
        indice_class = loss_class.argsort()[: choice_num]
        indice = sample_indice[indice_class]
        total_indice += indice.tolist()
    # 统计构建出的clean pool 中还混有污染样本的数量
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count+=1
    total_indice = np.array(total_indice)
    clean_pool_flag[total_indice] = 1

    print(
        "{}/{} poisoned samples in clean data pool".format(poisoned_count, clean_pool_flag.sum())
    )
    return clean_pool_flag

def class_agnostic_loss_guided_split(record_list, ratio, poisoned_indice):
    """
    Adaptively split the poisoned dataset by class-agnostic loss-guided split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    # poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_flag = np.zeros(len(loss))
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

def meta_split(record_list, meta_record_list, ratio, poisoned_indice):
    """
    Adaptively split the poisoned dataset by meta-split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    meta_loss = meta_record_list[keys.index("loss")].data.numpy()
    # poison = record_list[keys.index("poison")].data.numpy()
    clean_pool_flag = np.zeros(len(loss))
    loss = loss - meta_loss
    total_indice = loss.argsort()[: int(len(loss) * ratio)]
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count += 1
    print("{}/{} poisoned samples in clean data pool".format(poisoned_count, len(total_indice)))
    clean_pool_flag[total_indice] = 1
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

if __name__ == "__main__":
    main()