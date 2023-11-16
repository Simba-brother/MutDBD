import random
import sys
sys.path.append("./")

import torch
from torch.utils.data import DataLoader # 用于批量加载训练集的
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D,InputLayer
import cv2
import numpy as np

import onnx
from onnx2keras import onnx_to_keras

from codes.core.models.resnet import ResNet


def test_1():
    # 加载trained_pth_model
    model_pth = ResNet(18)
    # 加载权重
    model_pth.load_state_dict(torch.load("codes/models/ckpt_epoch_10.pth"))
    model_pth = model_pth.to(memory_format=torch.channels_last)
    input = torch.randn(1, 3, 32, 32)
    input = input.contiguous(memory_format=torch.channels_last)
    # 转换为ONNX
    torch.onnx.export(model_pth,               # model being run
                  input, # dummy input (required)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,
                  input_names=["input"],        # 输入名
                  output_names=["output"]    # 输出名
                ) # store the trained parameter weights inside the model file

    # Load the ONNX model
    model_onnx = onnx.load("resnet18.onnx")
    model_keras = onnx_to_keras(model_onnx, input_names = ['input'], change_ordering=True)
    # layers = model_keras.layers
    # for layer in layers:
    #     if hasattr(layer, "data_format") and layer.data_format == "channels_first":
    #         layer.data_format = "channels_last"
    model_keras.compile(loss='categorical_crossentropy', optimizer='adam')

    # 加载数据集
    device = torch.device("cpu")
    def _seed_worker():
        # 随机数种子
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

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
    
    # image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> torch.Tensor -> network input
    # 数据转换器
    transform = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])
    # 测试集
    testset = DatasetFolder(
        root='dataset/cifar10/test', # 测试集分类文件夹
        loader=cv2.imread, # 文件加载器
        extensions=('png',), # 文件后缀名 
        transform=transform, # 数据转换器
        target_transform=None, 
        is_valid_file=None)
    test_loader = DataLoader(
        testset,
        batch_size=16,
        shuffle=False,
        # num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=_seed_worker
    )
    predicts = [] # 预测标签
    labels = [] # 真实标签
    losses = [] # 损失
    for batch in test_loader:
        # 拿到一个batch
        batch_img, batch_label = batch # batch_label:one-hot
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
       
        batch_img = batch_img.numpy()
        batch_label =  batch_label.numpy()
        batch_img = tf.transpose(batch_img,[0,2,3,1])
        batch_output = model_keras.predict(batch_img,batch_size=1)
        batch_output = torch.from_numpy(batch_output)
        batch_label = torch.from_numpy(batch_label)
        predicts.append(batch_output)
        labels.append(batch_label)

    predicts = torch.cat(predicts, dim=0) 
    labels = torch.cat(labels, dim=0) 
    top_1_acc, top_5_acc = accuracy(predicts,labels,topk=(1, 5))
    print("top_1_acc:{},top_5_acc:{},loss:{}".format(top_1_acc, top_5_acc)) # , losses.mean().item()
    
if __name__ == "__main__":
    test_1()