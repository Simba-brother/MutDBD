'''
从某个变异率下的变异模型中选择出与original backdoor model性能最接近的top 50个模型
'''
import os
import queue
import logging
import setproctitle
import joblib
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from codes import config
from codes.utils import priorityQueue_2_list
from codes.ourMethod.detect_suspicious_classes.get_suspicious import get_suspicious_classes_by_ScottKnottESD


def load_csv(rate):
    csv_path = os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(rate),
        "preLabel.csv"
        )
    df = pd.read_csv(csv_path)
    return df

def get_priority_q(df):    
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    GT_labels = df["GT_label"]
    preLabels_o = df["original_backdoorModel_preLabel"]
    report_o = classification_report(GT_labels,preLabels_o,output_dict=True,zero_division=0)
    for m_i in range(500):
        col_name = f"model_{m_i}"
        preLabel_m = df[col_name]
        report_m = classification_report(GT_labels,preLabel_m,output_dict=True,zero_division=0)
        acc_dif = abs(report_o["accuracy"] - report_m["accuracy"])
        item = (acc_dif, m_i)
        q.put(item)
    return q
    
def get_top_k(queue,top_k=50):
    priority_list = priorityQueue_2_list(queue)
    selected_m_i_list = [m_i for priority, m_i in  priority_list[0:top_k]]
    return selected_m_i_list

def get_class_dict(df,selected_m_i_list):
    data_dict = defaultdict(list)
    GT_labels =  df["GT_label"]
    for m_i in selected_m_i_list:
        col_name = f"model_{m_i}"
        preLabels_m = df[col_name]
        report = classification_report(GT_labels,preLabels_m,output_dict=True,zero_division=0)
        for class_i in range(config.class_num):
            data_dict[class_i].append(report[str(class_i)]["precision"])
    return data_dict

def main(measure_name):
    # 计算数据
    data_dict = {}
    for rate in config.fine_mutation_rate_list:
        df = load_csv(rate)
        q = get_priority_q(df)
        selected_m_i_list = get_top_k(q,top_k=50)
        class_dict = get_class_dict(df,selected_m_i_list)
        suspicious_classes_top,suspicious_classes_low = get_suspicious_classes_by_ScottKnottESD(class_dict)
        data_dict[rate] = {"top":suspicious_classes_top,"low":suspicious_classes_low}
    
    # 保存数据
    save_dir = os.path.join(
        config.exp_root_dir,
        "SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = f"SuspiciousClasses_SK_Top50_{measure_name}.data"
    save_file_path = os.path.join(save_dir,save_file_name)
    joblib.dump(data_dict,save_file_path)
    # 打印提示信息
    logging.debug(f"SuspiciousClasses结果保存在:{save_file_path}")


def see_res(measure_name):
    data_dir = os.path.join(
        config.exp_root_dir,
        "SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name
    )
    file_name = f"SuspiciousClasses_SK_Top50_{measure_name}.data"
    data_path = os.path.join(data_dir,file_name)
    data_dict = joblib.load(data_path)
    target_class_idx = config.target_class_idx
    for rate in config.fine_mutation_rate_list:
        print(f"rate:{rate}")
        top_class_list = data_dict[rate]["top"]
        low_class_list = data_dict[rate]["low"]
        if target_class_idx in top_class_list:
            print("Top_group")
            ranking_within_the_group = top_class_list.index(target_class_idx)
            print(f"ranking_within_the_group:{ranking_within_the_group}")
        if target_class_idx in low_class_list:
            print("Low_group")
            ranking_within_the_group = low_class_list.index(target_class_idx)
            print(f"ranking_within_the_group:{ranking_within_the_group}")
        if (target_class_idx not in top_class_list) and (target_class_idx not in low_class_list):
            print("Mid_group")

if __name__ == "__main__":
    # 进程名称
    measure_name = "precision" # precision|recall|f1-score|
    proctitle = f"SuspiciousClasses_SK_Top50_{measure_name}|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = f"SuspiciousClasses_SK_Top50_{measure_name}.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    # 主函数
    main(measure_name)
    # see_res(measure_name)
