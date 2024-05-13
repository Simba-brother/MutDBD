# ASD baseline mian file
import sys
sys.path.append("./")
import os
import random
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # 用于批量加载训练集的

from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision.datasets import DatasetFolder
from codes import config
from codes.core.attacks import BadNets
from codes.core.models.resnet import ResNet
from codes.ASD.loss import SCELoss, MixMatchLoss
from codes.ASD.semi import poison_linear_record, mixmatch_train,linear_test
from codes.ASD.dataset import MixMatchDataset
from codes.ASD.log import result2csv
from codes.utils import create_dir

def main():
    exp_root_dir = config.exp_root_dir
    dataset_name = config.dataset_name
    model_name = config.model_name
    attack_name = config.attack_name
    class_num = config.class_num
    # 加载数据集
    # 训练集transform    
    transform_train = Compose([
        # Convert a tensor or an ndarray to PIL Image
        ToPILImage(), 
        # 训练数据增强,随机水平翻转
        RandomHorizontalFlip(),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ToTensor()
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        ToTensor()
    ])
    dataset_dir = "/data/mml/backdoor_detect/dataset/cifar10"
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(dataset_dir, "test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # backdoor pattern
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    # victim model
    model = ResNet(num=18, num_classes=10)        

    global_seed = 666
    deterministic = True
    # cpu种子
    torch.manual_seed(global_seed)

    def _seed_worker(worker_id):
        np.random.seed(global_seed)
        random.seed(global_seed)

    badnets = BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.1,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic
    )
    # 被污染的训练集
    poisoned_train_dataset = badnets.poisoned_train_dataset
    # 数据集中被污染的实例id
    poisoned_ids = poisoned_train_dataset.poisoned_set
    # 数据集加载器
    poisoned_train_dataset_loader = DataLoader(
            poisoned_train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            )
    clean_test_dataset_loader = DataLoader(
            testset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
    poisoned_test_dataset = badnets.poisoned_test_dataset
    poisoned_test_dataset_loader = DataLoader(
            poisoned_test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction = "mean")
    # 设备
    device = torch.device("cuda:0")
    # 损失函数对象放到gpu上
    criterion.to(device)
    # split 损失函数
    alpha = 0.1
    beta = 1
    num_classes = class_num
    split_criterion = SCELoss(alpha, beta,num_classes )
    # 损失函数对象放到gpu上
    split_criterion.to(device)
    # semi 损失函数
    lambda_u = 15  
    rampup_length = 120 # same as epoches
    semi_criterion = MixMatchLoss(rampup_length, lambda_u)
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    # clean seed samples
    clean_data_info = {}
    all_data_info = {}
    for class_idx in range(num_classes):
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
    for epoch in range(120):
        print("===Epoch: {}/{}===".format(epoch + 1, 120))
        if epoch < 60:
            record_list = poison_linear_record(model, poisoned_train_dataset_loader, split_criterion, device)
            if epoch % 5 == 0 and epoch != 0:
                choice_num += 5
            print("Mining clean data by class-aware loss-guided split...")
            # 0表示在污染池,1表示在clean pool
            split_indice = class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_ids)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < 90:       
            record_list = poison_linear_record(model, poisoned_train_dataset_loader, split_criterion, device)
            print("Mining clean data by class-agnostic loss-guided split...")
            split_indice = class_agnostic_loss_guided_split(record_list, 0.5, poisoned_ids)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < 120:
            record_list = poison_linear_record(model, poisoned_train_dataset_loader, split_criterion, device)
            meta_virtual_model = deepcopy(model)
            param_meta = [
                            {'params': meta_virtual_model.linear.parameters()},
                            {'params': meta_virtual_model.classifier.layer4.parameters()}
                        ]
            
            meta_optimizer = torch.optim.Adam(param_meta, lr=0.015)
            meta_criterion = nn.CrossEntropyLoss(reduction = "mean")
            meta_criterion.to(device)
            for _ in range(1):
                train_the_virtual_model(
                                        meta_virtual_model=meta_virtual_model, 
                                        poison_train_loader=poisoned_train_dataset_loader, 
                                        meta_optimizer=meta_optimizer,
                                        meta_criterion=meta_criterion,
                                        device = device
                                        )      
            meta_record_list = poison_linear_record(meta_virtual_model, poisoned_train_dataset_loader, split_criterion, device)

            print("Mining clean data by meta-split...")
            split_indice = meta_split(record_list, meta_record_list, 0.5, poisoned_ids)

            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)  

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
            **semi_mixmatch
        )

        print("Test model on clean data...")
        clean_test_result = linear_test(
            model, clean_test_dataset_loader, criterion
        )

        print("Test model on poison data...")
        poison_test_result = linear_test(
            model, poisoned_test_dataset_loader, criterion
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
        result2csv(result, save_dir)

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
            best_epoch = epoch + 1
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
        data.to(device)
        target.to(device)
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