import os
import sys
sys.path.append("./")
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip

from codes import core

'''
全局遍历区
'''
# 加载backdoor model
# model = core.models.ResNet(18)
# model.load_state_dict(torch.load('experiments/train_poisoned_DatasetFolder-CIFAR10_2023-10-04_10:20:00/ckpt_epoch_10.pth'))
# model.load_state_dict(torch.load('experiments/train_clean_CIFAR10/ckpt_epoch_10.pth'))
# 加载mutated_model
model=torch.load('experiments/CIFAR10/ResNet18_mutated_models/model_mutated_2.pth')
# 加载
# 加载全部干净训练集
# image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> torch.Tensor -> network input
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = DatasetFolder(
    root='./dataset/cifar10/train',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
# 加载全部干净测试集
transform_test = Compose([
    ToTensor()
])
testset = DatasetFolder(
    root='./dataset/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)
total_testset = len(testset)
print("干净测试集总数: {}".format(total_testset))
# 训练设备
device = torch.device("cpu")

# 加载全部中毒测试集 path:.core/attacks/BadNets.py
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0
badnets = core.BadNets(   # 后门方法，BadNets
    train_dataset=trainset, 
    test_dataset=testset,
    model=core.models.ResNet(18), # 神经网络模型
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1, # 攻击目标类
    poisoned_rate=0.05, # 污染率
    pattern=pattern, 
    weight=weight,
    # poisoned_transform_index=0,
    poisoned_transform_train_index=0,
    poisoned_transform_test_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666
)
# 得到纯污染测试数据集
poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
poisoned_index_set = poisoned_test_dataset.poisoned_set

poisoned_data_list = []
poisoned_taget_label_list = []

for index in poisoned_index_set:
    x, y, path, poisoned_flag = poisoned_test_dataset[index]
    if poisoned_flag is True:
        poisoned_data_list.append(x)
        poisoned_taget_label_list.append(y)

class PurePoisonedDataset(Dataset):
    
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        sample = self.data_list[index]
        label = self.label_list[index]
        return sample, label
    
purePoisonedDataset_test = PurePoisonedDataset(poisoned_data_list, poisoned_taget_label_list)
total_purePoisoned_testset = len(purePoisonedDataset_test)
print("纯污染的测试集总数: {}".format(total_purePoisoned_testset))

'''
方法过程区
'''
def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def caculate_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) # https://blog.csdn.net/mystyle_/article/details/111242489
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def predict(dataset):
    start_time = time.time()
    # 分批次
    batch_size = 16
    test_loss = nn.CrossEntropyLoss()
    # torch 无梯度环境
    with torch.no_grad():
        test_loader = DataLoader(
            dataset = dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
        # 模型评估模式
        model.eval()
        predicts = []
        labels = []
        losses = []
        for batch in test_loader:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            batch_predict = model(batch_img)
            loss = test_loss(batch_predict, batch_label)

            predicts.append(batch_predict.cpu()) # (B, self.num_classes)
            labels.append(batch_label.cpu()) # (B)
            if loss.ndim == 0: # scalar
                loss = torch.tensor([loss])
            losses.append(loss.cpu()) # (B) or (1)

        predicts = torch.cat(predicts, dim=0) # (N, self.num_classes)
        labels = torch.cat(labels, dim=0) # (N)
        losses = torch.cat(losses, dim=0) # (N)
        last_time = time.time()
        cost_time = int(last_time-start_time)
        return predicts, labels, losses.mean().item(), cost_time

def process_1():
    '''
    model在clean test set 预测结果
    '''
    predicts, labels, loss_mean, cost_time = predict(testset)
    prec1, prec5 = caculate_acc(predicts, labels, topk=(1,5))
    top1_correct = int(round(prec1.item() / 100.0 * total_testset))
    top5_correct = int(round(prec5.item() / 100.0 * total_testset))
    msg = "==========Test result on clean test dataset==========\n" + \
        f"Top-1 correct / Total: {top1_correct}/{total_testset}, Top-1 accuracy: {top1_correct/total_testset}, Top-5 correct / Total: {top5_correct}/{total_testset}, Top-5 accuracy: {top5_correct/total_testset}, mean loss: {loss_mean}, time: {cost_time}s\n"
    print(msg)

def process_2():
    '''
    model在pure poisoned test set 预测结果
    '''
    predicts, labels, loss_mean, cost_time = predict(purePoisonedDataset_test)
    prec1, prec5 = caculate_acc(predicts, labels, topk=(1,5))
    top1_correct = int(round(prec1.item() / 100.0 * total_purePoisoned_testset))
    top5_correct = int(round(prec5.item() / 100.0 * total_purePoisoned_testset))
    msg = "==========Test result on clean test dataset==========\n" + \
        f"Top-1 correct / Total: {top1_correct}/{total_purePoisoned_testset}, Top-1 accuracy: {top1_correct/total_purePoisoned_testset}, Top-5 correct / Total: {top5_correct}/{total_purePoisoned_testset}, Top-5 accuracy: {top5_correct/total_purePoisoned_testset}, mean loss: {loss_mean}, time: {cost_time}s\n"
    print(msg)


if __name__ == "__main__":
    # process_1()
    process_2()
    pass