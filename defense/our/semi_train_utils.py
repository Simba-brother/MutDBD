
from itertools import cycle,islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils.model_eval_utils import EvalModel

class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
        labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
            set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = dataset
        self.labeled = labeled
        if labeled:
            # 有标签的情况，从semi_id array中找到对应的索引
            # 比如arr = np.array([1,0,1,0])
            # np.nonzero(arr==1)[0]就为np.array([0,2])
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            # np.nonzero(arr==0)[0]就为np.array([1,3])
            self.semi_indice = np.nonzero(semi_idx == 0)[0]

    def __getitem__(self, index):
        # index in [0,len(self.semi_indice)-1]
        if self.labeled:
            item1 = self.dataset[self.semi_indice[index]] # self.semi_indice[index] = sampl_id(datset)
            img = item1[0]
            target = item1[1]
            item = {}
            item["img"] = img
            item["target"] = target
            item["labeled"] = True
        else:
            item1 = self.dataset[self.semi_indice[index]]
            item2 = self.dataset[self.semi_indice[index]]
            img1 = item1[0]
            img2 = item2[0]
            item = {}
            item["img1"] = img1
            item["img2"] = img2
            item["target"] = item1[1]
            item["labeled"] = False
        return item

    def __len__(self):
        # 这里的semi_indice其实就时选择出的带标签或不带标签的样本索引array
        return len(self.semi_indice)

class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        # epoch越接近epoch_total，越解决初始lambda
        if self.rampup_length == 0:
            return 1.0
        else:
            # 在迭代轮次初期 lambda_u会比较小
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch, class_weight = None):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        if class_weight is not None:
            Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget * class_weight, dim=1))
        else:
            Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]

def mixmatch_train_one_epoch(model, xloader, uloader, criterion, optimizer, epoch, device, **kwargs):

    # 数据加载器转化成迭代器
    xiter = cycle(xloader) # 有监督
    uiter = cycle(uloader) # 无监督
    xlimited_cycled_data = islice(xiter,0,kwargs["train_iteration"])
    ulimited_cycled_data = islice(uiter,0,kwargs["train_iteration"])
    model.train()
    
    for batch_idx,(xbatch,ubatch) in enumerate(zip(xlimited_cycled_data,ulimited_cycled_data)):
        xinput, xtarget = xbatch["img"], xbatch["target"]
        uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        
        # before_calcu_loss_start_time = time.perf_counter()
        batch_size = xinput.size(0)
        xtarget = torch.zeros(batch_size, kwargs["num_classes"]).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )

        # batch数据放到gpu上
        xinput = xinput.to(device)
        xtarget = xtarget.to(device) 
        uinput1 = uinput1.to(device)
        uinput2 = uinput2.to(device)
        # uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / kwargs["temperature"])
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()


        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(kwargs["alpha"], kwargs["alpha"])
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)
        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / kwargs["train_iteration"],
        )
        loss = Lx + lambda_u * Lu
        optimizer.zero_grad() # 优化器梯度清0
        loss.backward() # 基于loss函数求梯度值
        optimizer.step() # 优化器优化基于损失函数求的梯度优化模型参数

def semi_train(
        model,
        device,
        class_num,
        clean_seed, 
        epochs,
        lr,
        poisoned_trainset,
        labeled_id_set,
        unlabeled_id_set,
        all_id_list):
    
    split_indice = []
    for id in all_id_list:
        if id in labeled_id_set:
            split_indice.append(1)
        elif id in unlabeled_id_set:
            split_indice.append(0)
    assert len(split_indice) == len(all_id_list), "半监督数据切分不对"

    split_array = np.array(split_indice)
    xdata = MixMatchDataset(poisoned_trainset, split_array, labeled=True)
    udata = MixMatchDataset(poisoned_trainset, split_array, labeled=False)

    # 开始半监督训练
    # 开始clean pool进行监督学习,poisoned pool进行半监督学习
    xloader = DataLoader(xdata, batch_size=64, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    uloader = DataLoader(udata, batch_size=64, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    # semi 损失函数 # rampup_length = 120  same as epoches
    semi_criterion = MixMatchLoss(rampup_length=epochs, lambda_u=15)
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    semi_mixmatch = {"train_iteration": 1024,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num}

    
    best_model = model
    best_clean_seed_acc = 0
    for epoch in range(epochs):
        mixmatch_train_one_epoch(
            model,
            xloader,
            uloader,
            semi_criterion,
            optimizer,
            epoch,
            device,
            **semi_mixmatch
        )
        e = EvalModel(model,clean_seed,device)
        clean_seed_acc = e.eval_acc()
        print(f"epoch:{epoch},clean_seed_acc:{clean_seed_acc}")
        if clean_seed_acc > best_clean_seed_acc:
            best_model = model
    last_model = model
    return last_model, best_model