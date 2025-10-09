'''
一些小的工具函数和类
攻击函数：时间格式(format_time)，进度条，创建目录，熵计算(entropy)，优先级队列tolist(priorityQueue_2_list)，计算标签变化率(calcu_LCR)
类：Log
'''
import os
import torch
import numpy as np
import math
import random
import torch.nn
import time
import sys
import queue
import shutil
import logging
import sys
from collections import defaultdict
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
# _, term_width = os.popen('stty size', 'r').read().split()
_, term_width = shutil.get_terminal_size()
term_width = int(term_width)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def create_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
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

def priorityQueue_2_list(q:queue.PriorityQueue):
    qsize = q.qsize()
    res = []
    while not q.empty():
        res.append(q.get())
    assert len(res) == qsize, "队列数量不对"
    return res

def calcu_LCR(label_list_o:list,label_list:list):
    res = 0 
    count = 0
    for label_o,label in zip(label_list_o,label_list):
        if label_o != label:
            count += 1
    res = round(count/len(label_list_o),4)
    return res

class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

# 自定义异常钩子
def my_excepthook(exctype, value, traceback):
    logging.critical("Uncaught exception", exc_info=(exctype, value, traceback))
    # 调用默认的异常钩子，以防程序意外退出
    sys.__excepthook__(exctype, value, traceback)

def convert_to_hms(seconds):
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds = remaining_seconds % 60
    return hours, minutes, seconds

def nested_defaultdict(depth, default_factory=int):
    if depth == 1:
        return defaultdict(default_factory)
    else:
        return defaultdict(lambda: nested_defaultdict(depth - 1, default_factory))

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

if __name__ == "__main__":
    
    print()
    # makdir("experiments/test")