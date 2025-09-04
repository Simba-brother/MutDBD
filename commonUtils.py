import yaml
from scipy.stats import wilcoxon,mannwhitneyu,ks_2samp
from cliffs_delta import cliffs_delta
import logging
import os
import torch
import numpy as np
import random
import time
import sys
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
def get_class_num(dataset_name):
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    else:
        raise ValueError("Invalid input")
    return class_num

def compare_WTL(our_list, baseline_list,expect:str, method:str):
    ans = ""
    # 计算W/T/L
    # Wilcoxon:https://blog.csdn.net/TUTO_TUTO/article/details/138289291
    # Wilcoxon：主要来判断两组数据是否有显著性差异。
    if method == "wilcoxon": # 配对
        statistic, p_value = wilcoxon(our_list, baseline_list) # statistic:检验统计量
    elif method == "mannwhitneyu": # 不配对
        statistic, p_value = mannwhitneyu(our_list, baseline_list) # statistic:检验统计量
    elif method == "ks_2samp":
        statistic, p_value = ks_2samp(our_list, baseline_list) # statistic:检验统计量
    # 如果p_value < 0.05则说明分布有显著差异
    # cliffs_delta：比较大小
    # 如果参数1较小的话，则d趋近-1,0.147(negligible)
    d,res = cliffs_delta(our_list, baseline_list)
    if p_value >= 0.05:
        # 值分布没差别
        ans = "Tie"
        return ans
    else:
        # 值分布有差别
        if expect == "small":
            # 指标越小越好，d越接近-1越好
            if d < 0 and res != "negligible":
                ans = "Win"
            elif d > 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
        else:
            # 指标越大越好，d越接近1越好
            if d > 0 and res != "negligible":
                ans = "Win"
            elif d < 0 and res != "negligible":
                ans = "Lose"
            else:
                ans = "Tie"
    return ans

def get_logger(log_dir,log_file_name):
    # 创建一个logger
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

def set_random_seed(random_seed):
    # cpu种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

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
