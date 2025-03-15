import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from codes import utils

def random_seed():
    worker_seed = 666
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def adjust_learning_rate(optimizer, init_lr, epoch):
    # epoch: 当前训练进行时的epoch
    # step: batch_id
    # len_epoch: epoch中有多少个batch ,也就是每轮次（epoch）中迭代了多少步（step）,注意: 此处为向上取整
    
    # 统计小于当前epoch的个数作为因子
    factor = (torch.tensor([150,180]) <= epoch).sum()
    lr = init_lr*(0.1**factor)

    # """Warmup"""
    # if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
    #     lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)

    # 对优化器中的学习率进行赋值更新
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

def train(model, trainset, epochs, batch_size, optimizer, init_lr, loss_fn, device, work_dir, scheduler):
    trainset_loader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle=True,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=random_seed()
        )
    best_acc = 0
    best_model = None
    model.to(device)
    model.train()
    record_acc = []
    record_loss = []
    for epoch in range(epochs):
        print('Epoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainset_loader):
            adjust_learning_rate(optimizer, init_lr, epoch)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() # 每个batch的累计损失
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # utils.progress_bar(batch_idx, len(trainset_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        epoch_acc = round(correct/total,3)
        epoch_avg_batch_loss =  round(train_loss / (batch_idx+1),3)
        record_acc.append(epoch_acc)
        record_loss.append(epoch_avg_batch_loss)
        print(f"epoch_trainset_Acc:{epoch_acc}")
        if epoch_acc > best_acc:
            best_model = model
            best_acc = epoch_acc
            utils.create_dir(work_dir)
            ckpt_model_path = os.path.join(work_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_model_path)
            print(f"best model is saved in {ckpt_model_path}")
        if scheduler is not None:
            scheduler.step()
    ans = {
        "best_model":best_model,
        "best_acc":best_acc,
        "record_acc":record_acc,
        "record_loss":record_loss
    }
    return ans


def test(model,testset,batch_size,device,loss_fn):
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn= random_seed()
        )
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testset_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # utils.progress_bar(batch_idx, len(testset_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = round(correct/total,3)
    return acc