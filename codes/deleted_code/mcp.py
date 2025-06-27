import sys
sys.path.append("./")
import numpy as np
import random
import queue
import cv2
import torch
import copy
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

from core.models.resnet import ResNet
from codes.common.eval_model import EvalModel
from bigUtils import priorityQueue_2_list, create_dir
from codes.tools.draw import draw_line
def _seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
model = ResNet(18)
model.load_state_dict(torch.load("/data/mml/backdoor_detect/experiments/CIFAR10/resnet18_nopretrain_32_32_3/clean/best_model.pth", map_location="cpu"))
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
trainset = DatasetFolder(
        root='/data/mml/backdoor_detect/dataset/cifar10/train',
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

testset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/test',
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

device = torch.device("cuda:0")
e = EvalModel(model, testset, device)
prob_outputs = e._get_prob_outputs()
p_label_list = e._get_pred_labels()
true_false_list = e._get_TrueOrFalse()

truecount = sum(true_false_list)
falsecount = len(true_false_list)-truecount

def get_mcp_matrix(prob_outputs, true_false_list):
    matrix = [[None for i in range(10)] for j in range(10)]
    for i in range(10):
        for j in range(10):
            q = queue.PriorityQueue()
            matrix[i][j] = q
    for i in range(len(prob_outputs)):
        prob_list = prob_outputs[i]
        sorted_prob_list = sorted(prob_list)
        max_p = sorted_prob_list[-1]
        second_p = sorted_prob_list[-2]
        priority = second_p/max_p # 值越大优先级越高
        label_one = prob_list.index(max_p)
        label_two = prob_list.index(second_p)
        flag = true_false_list[i]
        item = (-priority, label_one, flag)
        matrix[label_one][label_two].put(item)
    # total = 0
    # for i in range(10):
    #     for j in range(10):
    #         total += matrix[i][j].qsize()
    # assert total == len(prob_outputs), "数目不对"
    return matrix



cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

precision_list = []
recall_list = []
for cut_off in cut_off_list:
    matrix = get_mcp_matrix(prob_outputs, true_false_list)
    target_num = int(len(prob_outputs)*cut_off)
    prefix_priority_list = []
    while target_num > 0:
        for i in range(10):
            for j in range(10):
                if matrix[i][j].qsize() == 0:
                    continue
                prefix_priority_list.append(matrix[i][j].get()) 
                target_num -= 1
    TP = 0
    FP = 0
    gt_TP = falsecount
    for item in prefix_priority_list:
        gt_label = item[2]
        if gt_label is False:
            TP += 1
        else:
            FP += 1
    precision = round(TP/(TP+FP),3)
    recall = round(TP/gt_TP,3)
    precision_list.append(precision)
    recall_list.append(recall)
    print("FP:",FP)
    print("TP:",TP)
    print("precision:",precision)
    print("recall:",recall)
    print("truecount:", truecount)
    print("falsecount:", falsecount)
y = {"precision":precision_list, "recall":recall_list}
title = "mcp"
save_path = "mcp"
draw_line(cut_off_list, title, save_path, **y)
