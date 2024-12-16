'''
检测target class
'''
import os
from codes import config
import torch
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
import numpy as np
from scipy import stats
from cliffs_delta import cliffs_delta
import joblib
import logging
import setproctitle


def calu_p_and_dela_value(list_1, list_2):
    p_value = stats.wilcoxon(list_1, list_2).pvalue
    list_1_sorted = sorted(list_1) # 原来list不改变
    list_2_sorted = sorted(list_2)
    delta,info = cliffs_delta(list_1_sorted, list_2_sorted)
    return p_value,delta


def get_target_class(data_dict):
    '''
    data_dict:{Int(class_idx):list(precision|recall|F1)}
    '''
    target_class = -1
    max_avg = -1
    max_avg_class_idx = -1
    for i in range(config.class_num):
        array = np.array(data_dict[i])
        # 当前类别i得指标均值
        avg_value = np.round(np.mean(array),decimals=4).item()
        # 当前类别i得指标与其他指标的Wilcoxon rank sum test p值
        p_list = []
        # 当前类别i得指标与其他指标的Cliff’s delta 值
        delta_list = []
        for j in range(config.class_num):
            if j == i:
                continue
            p_value,delta = calu_p_and_dela_value(data_dict[i],data_dict[j])
            p_list.append(p_value)
            delta_list.append(delta)
        all_P_flag = all(p_value < 0.05 for p_value in p_list)
        all_C_flag = all(d > 0.147 for d in delta_list)
        if all_P_flag and all_C_flag:
            # i类别指标分布与其他类别显著有区别且值较大
            target_class = i
            return target_class
        if avg_value > max_avg:
            max_avg = avg_value
            max_avg_class_idx = i
    return max_avg_class_idx






def reconstruct_data(report_dataset,measure_name):
    '''
    args:
        report_dataset:
            {
                ratio:{
                    operation:[report_classification]
                }
            }
        measure_name:str,precision|recall|f1
    return:
        {
            ratio:{
                class_id:[measure]
            }
        }
    '''
    data = {}
    for ratio in config.fine_mutation_rate_list:
        data[ratio] = {}
        for op in config.mutation_name_list:
            for class_i in range(config.class_num):
                data[ratio][class_i] = []
                for report in report_dataset[ratio][op]:
                    data[ratio][class_i].append(report[str(class_i)][measure_name])
    return data

def detect(report_dataset):
    '''
    args:
        report_dataset:
            {
                ratio:{
                    operation:[report_classification]
                }
            }
    return:
        {
            ratio:target class
        }
    '''
    ans = {}
    data = reconstruct_data(report_dataset,measure_name="precision")
    for ratio in config.fine_mutation_rate_list:
        target_class = get_target_class(data[ratio])
        ans[ratio] = target_class
    return ans



if __name__ == "__main__":
    # 进程名称
    proctitle = f"TargetClass|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device("cuda:0")

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "TargetClass.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    # 加载变异模型评估结果
    evalMutationResult = joblib.load(os.path.join(
        config.exp_root_dir,
        "EvalMutationResult",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "EvalMutationResult.data"
    ))
    # 得到各个变异率下的target class
    target_class_ans = detect(evalMutationResult)
    # 保存实验结果
    save_dir = os.path.join(
        config.exp_root_dir,
        "TargetClass",
        config.dataset_name, 
        config.model_name, 
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "TargetClass.data"
    save_file_path = os.path.join(save_dir,save_file_name)
    joblib.dump(target_class_ans,save_file_path)
    logging.debug(f"target class结果保存在:{save_file_path}")


    