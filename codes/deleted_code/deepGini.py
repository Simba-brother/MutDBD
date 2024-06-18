import sys
sys.path.append("./")
import numpy as np
import random
import queue
import cv2
import torch
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

from core.models.resnet import ResNet
from codes.tools.eval_model import EvalModel
from utils import priorityQueue_2_list, create_dir
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
true_false_list = e._get_TrueOrFalse()
truecount = sum(true_false_list)
falsecount = len(true_false_list)-truecount

def caculate_deepGini(prob_outputs):
    deepGini_list = []
    for i in range(len(prob_outputs)):
        prob_list = prob_outputs[i]
        sum = 0
        for p in prob_list:
            sum += p*p
        deepgini= 1-sum
        deepGini_list.append(deepgini)
    return deepGini_list
def get_queue(deepGini_list, true_false_list):
    q = queue.PriorityQueue()
    for deepGini, flag in zip(deepGini_list, true_false_list):
        item = (-deepGini, flag)
        q.put(item)
    return q

deepGini_list = caculate_deepGini(prob_outputs)
q = get_queue(deepGini_list, true_false_list)
priority_list = priorityQueue_2_list(q)
cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
precision_list = []
recall_list = []
for cut_off in cut_off_list:
    end = int(len(priority_list)*cut_off)
    prefix_priority_list = priority_list[0:end]
    TP = 0
    FP = 0
    gt_TP = falsecount
    for item in prefix_priority_list:
        gt_label = item[1]
        if gt_label == False:
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
title = "DeepGini"

save_path = "DeepGini.png"
draw_line(cut_off_list, title, save_path, **y)
