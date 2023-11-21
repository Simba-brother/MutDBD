import os
import torch
import numpy as np
import math
import random
import torch.nn

def create_dir(dir_path):
    if os.path.exists(dir_path):
        print(f"文件夹{dir_path}已经存在!")
    else:
        print(f"成功创建文件夹{dir_path}")
        os.makedirs(dir_path)


def random_seed():
    # worker_seed = torch.initial_seed() % 2**32
    worker_seed = 666
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    os.environ['PYTHONHASHSEED'] = str(worker_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def entropy(data):
    """
    计算信息熵
    :param data: 数据集
    :return: 信息熵
    """
    length = len(data)
    counter = {}
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

class ModelReview(object):
    def __init__(self):
        self.model = None
    def set_model(self, model):
        self.model = model
    def get_model(self, model):
        return self.model
    def see_layers(self):
        model = self.model
        layers = [module for module in model.modules()]
        print(f"总共层数:{len(layers)}")
        print("="*20)
        for layer in layers:
            print(layer,"\n")
            # print(isinstance(layer, torch.nn.Linear))
        print("="*20)
if __name__ == "__main__":

    pass
    # makdir("experiments/test")