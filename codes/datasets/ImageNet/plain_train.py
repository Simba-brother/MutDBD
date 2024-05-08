import sys
sys.path.append("./")
import random
import time
import shutil
import joblib
import os
import cv2
import numpy as np
from enum import Enum

import torch
# import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
# import torch.distributed as dist
# from torch.utils.data import Subset
from torchvision.models import resnet18,vgg19,densenet121
import setproctitle
from torchvision.datasets import DatasetFolder
# from codes.core.models.resnet import ResNet
# from codes.scripts.dataset_constructor import ExtractDatasetAndModifyLabel
from codes.utils import create_dir
from codes import config

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # val: 当前计算出的值
        self.val = val
        # 把当前val加到对象的sum属性上
        self.sum += val * n
        # 把统计的次数加到对象的count属性上
        self.count += n
        # 计算avg并赋值到self.avg属性上
        self.avg = self.sum / self.count

    # def all_reduce(self):
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     elif torch.backends.mps.is_available(): # 于检查当前 PyTorch 版本是否支持使用 NVIDIA MPS (Multi-Process Service)。
    #         device = torch.device("mps")
    #     else:
    #         device = torch.device("cpu")
    #     total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
    #     dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    #     self.sum, self.count = total.tolist()
    #     self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # _为values,pred为indices
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def train(train_loader, model, criterion, optimizer, epoch, device):
    '''一个epoch的训练'''
    print_freq = 10
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f') # 一批次数据加载的时间
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
     # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # current batch
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device)
        target = targets.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad() # 剔除以前的导数值
        loss.backward() # 求导
        optimizer.step() # 参数更新

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i + 1)

def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    base_progress = 0
    print_freq = 10
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            i = base_progress + i
            images = images.to(device)
            targets = targets.to(device)
            # compute output
            output = model(images)
            loss = criterion(output, targets)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                progress.display(i + 1)
    progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    '''
    保存检查点
    args:
        state:{
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1':  best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }
        is_best:当前检查点是否是最好的。
    '''
    save_dir = os.path.join(config.exp_root_dir,config.dataset_name, config.model_name, "plain_train")
    create_dir(save_dir)
    save_file_path = os.path.join(save_dir, filename)
    torch.save(state, save_file_path)
    if is_best:
        shutil.copyfile(save_file_path, os.path.join(save_dir, "model_best.pth"))


def extract_dataset_to_dir():
    # ImageNet_2012 task 1 3个下载地址
    # https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    # https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    # https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
    # 将下载好的压缩文件放到一个目录下,则下面代码会将数据抽取到文件夹中并使用文件夹形式分类。
    # 从 tar 抽取数据集 到 train 和 val
    # train_dataset = torchvision.datasets.ImageNet(root=ImageNet_dataset_dir, split='train',transform=transform_train)
    # val_dataset = torchvision.datasets.ImageNet(root=ImageNet_dataset_dir, split='val',transform=transform_val)

    # 抽取子集
    # target_set = set(train_dataset.targets)
    # sampling_targets(target_set, 30)
    # sampled_targets_and_remapp = joblib.load(os.path.join(config.exp_root_dir, config.dataset_name, "sampled_targets_and_remapp.data"))
    # sampled_targets = sampled_targets_and_remapp["sampled_targets"]
    # target_remapp = sampled_targets_and_remapp["target_remapp"]
    # train_sample_indices = [i for i, target in enumerate(train_dataset.targets) if target in sampled_targets]
    # sample_train_dataset = Subset(train_dataset,train_sample_indices)
    # val_sample_indices = [i for i, target in enumerate(val_dataset.targets) if target in sampled_targets]
    # sample_val_dataset = Subset(val_dataset,val_sample_indices)
    # sample_train_dataset = ExtractDatasetAndModifyLabel(sample_train_dataset, target_remapp)
    # sample_val_dataset = ExtractDatasetAndModifyLabel(sample_val_dataset, target_remapp)
    return

def sampling_targets(targets, sampled_num = 30):
    # random.sample函数是一个无放回的抽样，这意味着每个元素只能被选择一次
    sampled_targets = random.sample(targets, sampled_num)
    sampled_targets.sort()
    target_remapp  = {}
    for i in range(len(sampled_targets)):
        sampled_target = sampled_targets[i]
        target_remapp[sampled_target] = i
    # 保存数据
    save_dir = "/data/mml/backdoor_detect/experiments/ImageNet"
    create_dir(save_dir)
    save_file_name =  "sampled_targets_and_remapp.data"
    save_file_path = os.path.join(save_dir, save_file_name)
    data = {"sampled_targets":sampled_targets,"target_remapp":target_remapp}
    joblib.dump(data, save_file_path)
    print("sampling_targets() success")
    print(f"save_file_path:{save_file_path}")
    
def get_subset():
    train_dir = "/data/mml/dataset/ImageNet_2012/train"
    val_dir = "/data/mml/dataset/ImageNet_2012/val"
    class_dir_list = os.listdir(train_dir)
    sampled_class_list = random.sample(class_dir_list, 30)
    for sampled_class in sampled_class_list:
        sampled_class_dir_path = os.path.join(train_dir, sampled_class)
        source_dir_path = sampled_class_dir_path
        target_dir_path = os.path.join("/data/mml/dataset/ImageNet_2012","subset","train",sampled_class)
        shutil.copytree(source_dir_path, target_dir_path)
        

        sampled_class_dir_path = os.path.join(val_dir, sampled_class)
        source_dir_path = sampled_class_dir_path
        target_dir_path = os.path.join("/data/mml/dataset/ImageNet_2012","subset","val",sampled_class)
        shutil.copytree(source_dir_path, target_dir_path)
    print("get_subset end")

def main_worker():
    ImageNet_subset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"
    transform_train = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std = [ 0.229, 0.224, 0.225 ]),
    ])
    transform_val = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std = [ 0.229, 0.224, 0.225 ]),
    ])

    trainset = DatasetFolder(
        root= os.path.join(ImageNet_subset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None
        )
    valset = DatasetFolder(
        root= os.path.join(ImageNet_subset_dir, "val"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('jpeg',),
        transform=transform_val,
        target_transform=None,
        is_valid_file=None)

    # 数据加载batch_size
    batch_size = 256
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)
    # victim model
    model = densenet121(pretrained = True)
    config.model_name = "DensNet"
    setproctitle.setproctitle("ImageNet|DensNet|plain_train")
    # 冻结预训练模型中所有参数的梯度
    for param in model.parameters():
        param.requires_grad = False
    # 修改最后一个全连接层的输出类别数量
    num_classes = 30  # 假设我们要改变分类数量为30
    '''
    ResNet18
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    VGG19
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    Densnet121
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    '''
    
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    device = torch.device("cuda:1")
    model.to(device)
    # 交叉熵损失函数对象放到设备上
    criterion = nn.CrossEntropyLoss().to(device)
    # SGD优化器
    optimizer = torch.optim.SGD(
        model.parameters(), # model 参数
        lr = 0.1,
        momentum = 0.9,
        weight_decay=1e-4)
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # 训练前使用valset评估一下模型
    validate(val_loader, model, criterion, device)
    best_acc1 = 0
    # 训练的轮次
    epoches = 10
    for epoch in range(epoches):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)
        # evaluate on validation set of an epoch
        acc1 = validate(val_loader, model, criterion, device)
        # lr step走一步
        scheduler.step()
        # 当前轮次训练结果是最好的:bool
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
                    'epoch': epoch + 1, # 下次开始的epoch
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best)
    print("main_worker() end")




if __name__ == "__main__":
    '''全局变量区'''
    global_seed = 666
    random.seed(global_seed)
    np.random.seed(global_seed)
    deterministic = True
    # cpu种子
    torch.manual_seed(global_seed)
    main_worker()
    # get_subset()
