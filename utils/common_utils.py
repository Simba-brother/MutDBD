import yaml

import logging
import os
import torch
import numpy as np
import random
import time
import sys

class Record(object):
    '''
    一个批次一个批次的记录数据
    '''
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = torch.zeros(self.size)

    def update(self, batch_data):
        self.data[self.ptr : self.ptr + len(batch_data)] = batch_data
        self.ptr += len(batch_data)

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
def get_logger(log_dir,log_file_name):
    # 创建一个logger实例
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,log_file_name)

    # logger的文件处理器，包括日志等级，日志路径，模式，编码等
    file_handler = logging.FileHandler(log_path,mode="w",encoding="UTF-8")
    file_handler.setLevel(logging.DEBUG)

    # logger的格式化处理器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    #将格式化器添加到文件处理器
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志对象中
    logger.addHandler(file_handler)
    return logger

import sys, logging
sys.path.append('../../')

import random
import numpy as np
import torch
import torchvision.transforms as transforms


def set_random_seed(random_seed):
    '''
    设置运行时的随机种子
    '''
    random.seed(random_seed) # random 标准库
    np.random.seed(random_seed) # np.random 库
    torch.manual_seed(random_seed) # torch
    torch.cuda.manual_seed_all(random_seed) # torch.cuda
    torch.backends.cudnn.deterministic = True # torch.backends.cudnn
    torch.backends.cudnn.benchmark = False # # torch.backends.cudnn

def convert_to_hms(seconds):
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds = remaining_seconds % 60
    return hours, minutes, seconds

def get_formattedDateTime():
    '''
    用于生成格式化的时间
    '''
    timestamp = time.time()
    date_time = time.localtime(timestamp)
    formatted_time = time.strftime('%Y-%m-%d_%H:%M:%S', date_time)
    return formatted_time

def my_excepthook(exctype, value, traceback):
    logging.critical("Uncaught exception", exc_info=(exctype, value, traceback))
    # 调用默认的异常钩子，以防程序意外退出
    sys.__excepthook__(exctype, value, traceback)

def create_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        os.makedirs(dir_path)

