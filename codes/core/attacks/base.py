from copy import deepcopy
import math
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # 用于批量加载训练集的
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log

# 支持的数据集类型
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)


def check(dataset):
    return isinstance(dataset, support_list)


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


class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset
        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model 
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
            torch.use_deterministic_algorithms(True) 
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # 防止下行设置因为版本问题报错
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch):
        # epoch: 当前训练进行时的epoch
        # step: batch_id
        # len_epoch: epoch中有多少个batch ,也就是每轮次（epoch）中迭代了多少步（step）,注意: 此处为向上取整
        # self.train之后会注入self.current_schedule
        # 统计小于当前epoch的个数作为因子
        factor = (torch.tensor(self.current_schedule['schedule']) <= epoch).sum()

        lr = self.current_schedule['lr']*(self.current_schedule['gamma']**factor)

        """Warmup"""
        if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
            lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)
        # 对优化器中的学习率进行赋值更新
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, schedule=None):
        # schedule:type = dict
        # 为self.current_schedule注入schedule(计划表)
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)
        # 设置日志文件存储位置
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        # 注入work_dir属性
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        # 声明出一个log实例
        log = Log(osp.join(work_dir, 'log.txt'))
        self.log = log

        log('==========Schedule parameters==========\n')
        log(str(self.current_schedule)+'\n')

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain'], map_location='cpu'), strict=False)
            log(f"Load pretrained parameters: {self.current_schedule['pretrain']}\n")
        ''''
        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")
        '''
        device = torch.device(self.current_schedule['device'])
        self.model.to(device)
        log(f"device:{device}")
        if self.current_schedule['benign_training'] is True:
            # clean train
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=False,
                worker_init_fn=self._seed_worker
            )
        elif self.current_schedule['benign_training'] is False:
            # attack train
            train_loader = DataLoader(
                self.poisoned_train_dataset, # Base下边的子类（例如BadNets）的属性
                batch_size=self.current_schedule['batch_size'], # default:128
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=False,
                worker_init_fn=self._seed_worker
            )
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")

        # 模型进入训练模式
        self.model.train()
        # target model的参数优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule ['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        '''
        # 改为我自己benign 训练的优化器
        params_1x = [param for name, param in self.model.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam([{'params':params_1x}, {'params': self.model.fc.parameters(), 'lr': self.current_schedule['lr']*10}], lr=self.current_schedule['lr'], weight_decay=self.current_schedule['weight_decay'])
        '''
        # 总迭代次数，每个batch,iteration++
        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        best_acc = 0
        for i in range(self.current_schedule['epochs']):
            # 每次epoch都进入train mode
            self.model.train()
            # steps
            for batch_id, batch in enumerate(train_loader):
                # 每一轮中的批次
                # 动态调整 optimizer 中的学习率
                self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))
                batch_img = batch[0]
                batch_label = batch[1] # num
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                # forward
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                # backward
                optimizer.zero_grad()
                # 计算损失函数中参数导数
                loss.backward()
                # 优化器优化参数
                optimizer.step()
                # 参数更新次数
                iteration += 1
                # 记录一下训练信息到log文件
                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    last_time = time.time()
                    log(msg)

            # 每轮次过后，寻找下best model并保存
            predict_digits, labels, mean_loss = self._test(self.poisoned_train_dataset, device, self.current_schedule['batch_size'])
            # 总共预测样本的数量
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            save_path = os.path.join(work_dir, "best_model.pth")
            if prec1 > best_acc:
                best_acc = prec1
                self.best_model = self.model
                torch.save(self.model.state_dict(), save_path)
            msg = "==========Test result on poisoned_train_dataset==========\n" + \
            f"Epoch:{i+1}/{self.current_schedule['epochs']}, Top-1 accuracy: {prec1}"
            log(msg)
            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # 每经过5个epoch测试一下训练效果
                # test result on benign test dataset
                predict_digits, labels, mean_loss = self._test(self.test_dataset, device, self.current_schedule['batch_size'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                # 记录下测试结果
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)
                # test result on poisoned test dataset（ASR）
                predict_digits, labels, mean_loss = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)
                # 评估完后modelj进入train mode
                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                # 保存前进入评估模式
                self.model.eval()
                torch.save(self.model.state_dict(), ckpt_model_path)
                # 保存后赶紧训练模式
                self.model.train()
        # epochs完成后model放入cpu返回
        self.model.eval()
        self.model = self.model.cpu()

    def _test(self, dataset, device, batch_size=16, model=None, test_loss=None):
        '''
        测试一下self.model在dataset上的效果
        return:
            predict_digits: model在dataset的output
            labels: dataset的gt_label
            losses.mean().item(): output的损失标量
        '''
        if model is None:
            model = self.model
        else:
            model = model

        if test_loss is None:
            test_loss = self.loss # 损失函数：nn.CrossEntropyLoss()
        else:
            test_loss = test_loss

        # torch 无梯度环境
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
            # 模型评估模式
            model.eval()

            predict_digits = []
            labels = []
            losses = []
            for batch in test_loader:
                batch_img, batch_label = batch[0],batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                batch_img = model(batch_img)
                loss = test_loss(batch_img, batch_label)

                predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
                labels.append(batch_label.cpu()) # (B)
                if loss.ndim == 0: # scalar
                    loss = torch.tensor([loss])
                losses.append(loss.cpu()) # (B) or (1)

            predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
            labels = torch.cat(labels, dim=0) # (N)
            losses = torch.cat(losses, dim=0) # (N)
            return predict_digits, labels, losses.mean().item()

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None, test_loss=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        # 保存训练结果dir
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        # 创建出来个log
        log = Log(osp.join(work_dir, 'log.txt'))
        '''
        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")
        '''
        device = torch.device(self.current_schedule['device']) 
        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels, mean_loss = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels, mean_loss = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], model, test_loss)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

        return top1_correct, top5_correct, total_num, mean_loss
