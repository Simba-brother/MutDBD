'''
ResNet in PyTorch.
# 替代 https://blog.csdn.net/weixin_62894060/article/details/130718618
'''
import math
import time
import os
import sys
sys.path.append("./")
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize

from codes import utils


def _seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    gamma = 0.1
    lr = 0.1
    factor = (torch.tensor([150,180]) <= epoch).sum()
    lr = lr*(gamma**factor)
    # """Warmup"""
    # if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
    #     lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
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

def get_transform(flag):
    if flag == 1:
        transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
            ToPILImage(),
            Resize((32, 32)),
            ToTensor()
        ])

        transform_test = Compose([
            ToPILImage(),
            Resize((32, 32)),
            ToTensor()
        ])
    elif flag == 2:
        # transform
        transform_train=Compose([
            # Resize step is required as we will use a ResNet model, which accepts at leats 224x224 images
            ToPILImage(), # PIL.Image
            Resize((224,224)),  
            ToTensor() # C x H x W
        ])
        transform_test = Compose([
            ToPILImage(),
            Resize((224,224)), 
            ToTensor()
        ])
    else:
        raise KeyError("flag value error")
    return transform_train, transform_test

def start_train():
    work_dir = "experiments/CIFAR10/models/resnet18_nopretrain_32_32_3/clean"
    log_path = os.path.join(work_dir, "log.txt")
    log = utils.Log(log_path)
    # 获得model
    model = ResNet(num=18, num_classes=10)
    # 获得transform
    transform_train, transform_test = get_transform(flag = 1)
    # 获得数据集
    trainset = DatasetFolder(
        root='./dataset/cifar10/train',
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root='./dataset/cifar10/test',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # 获得加载器
    batch_size = 128
    trainset_loader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle=True,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
        )
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
        )
    # 获得参数优化器
    lr = 0.1
    momentum = 0.9
    weight_decay=5e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # 开始训练
    epoches = 200
    device = torch.device("cuda:4")
    loss_fn = nn.CrossEntropyLoss()
    log_iteration_interval = 100
    test_epoch_interval = 10
    save_epoch_interval = 10
    iteration = 0 # 总的迭代次数
    model.to(device)
    best_acc = 0
    last_time = time.time()
    for epoch_id in range(epoches):
        # 训练轮次
        for batch_id, batch in enumerate(trainset_loader):
            # 每一批次
            # 动态调整 optimizer 中的学习率
            steps = math.ceil(len(trainset) / batch_size) 
            adjust_learning_rate(optimizer, epoch_id, batch_id, steps)
            batch_img = batch[0]
            batch_label = batch[1]
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            predict_digits = model(batch_img)
            loss = loss_fn(predict_digits, batch_label)
            loss.backward()
            optimizer.step()
            # 参数更新次数
            iteration += 1
            if iteration % log_iteration_interval == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Epoch:{epoch_id+1}/{epoches}, iteration:{batch_id + 1}/{steps},"+ \
                f"lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                last_time = time.time()
                log(msg)

        # 每完成一轮训练评估一下, 保存best model of trainset
        predict_digits, labels, mean_loss = start_test(model, trainset, device, batch_size, loss_fn)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        if prec1 > best_acc:
            best_acc = prec1
            save_path = os.path.join(work_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        model.train()

        if (epoch_id + 1) % test_epoch_interval == 0:
            # 一定轮次区间后 测一下model性能
            predict_digits, labels, mean_loss = start_test(model, testset, device, batch_size, loss_fn)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)
            model.train()

        if (epoch_id + 1) % save_epoch_interval == 0:
            # 一定轮次区间后, 保存model ckpt
            ckpt_model_filename = "ckpt_epoch_" + str(epoch_id+1) + ".pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            model.eval()
            torch.save(model.state_dict(), ckpt_model_path)
            model.train()


def start_test(model, dataset, device, batch_size, loss_fn):
    # 模型评估模式
    model.eval()
    # torch 无梯度环境
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )
        
        predict_digits = []
        labels = []
        losses = []
        for batch in test_loader:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            batch_img = model(batch_img)
            loss = loss_fn(batch_img, batch_label)

            predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
            labels.append(batch_label.cpu()) # (B)
            if loss.ndim == 0: # scalar
                loss = torch.tensor([loss])
            losses.append(loss.cpu()) # (B) or (1)

        predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
        labels = torch.cat(labels, dim=0) # (N)
        losses = torch.cat(losses, dim=0) # (N)
        return predict_digits, labels, losses.mean().item()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # BasicBlock中的第1卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # BasicBlock中的第2卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet(num, num_classes=10):
    if num == 18:
        return _ResNet(BasicBlock, [2,2,2,2], num_classes)
    elif num == 34:
        return _ResNet(BasicBlock, [3,4,6,3], num_classes)
    elif num == 50:
        return _ResNet(Bottleneck, [3,4,6,3], num_classes)
    elif num == 101:
        return _ResNet(Bottleneck, [3,4,23,3], num_classes)
    elif num == 152:
        return _ResNet(Bottleneck, [3,8,36,3], num_classes)
    else:
        raise NotImplementedError



if __name__ == "__main__":
    # start_train()
    pass
