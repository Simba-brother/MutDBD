'''
怀疑集检测器
'''
import os
import queue
import json
import pandas as pd
from collections import defaultdict
from codes import config
# 得到格式化时间串
from sklearn.metrics import classification_report
from codes.common.time_handler import get_formattedDateTime
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids
from codes.utils import priorityQueue_2_list
from codes.common.logging_handler import get_Logger

'''
=======核心函数区==================
'''

def detect_by_loss():
    '''
    通过loss排名进行怀疑集检测
    '''
    pass

def detect_by_precision(df:pd.DataFrame,mutated_model_topk:int):
    '''
    通过precision排名进行怀疑集检测
    '''
    # 变异模型global_id_list
    mutated_model_global_id_list = None
    if mutated_model_topk == -1:
        # 全部变异模型
        mutated_model_global_id_list = list(range(500))
    else:
        mutated_model_global_id_list = get_top_k_global_ids(df,top_k=mutated_model_topk,trend="bigger")
    
    gt_label_list = df["GT_label"]
    '''
    dict(list[int])
    {class_id:[precision_1,..,]}
    '''
    class_precisionList_dict = defaultdict(list)
    for i in mutated_model_global_id_list:
        preLabel_list = df[f"model_{i}"]
        report = classification_report(gt_label_list,preLabel_list,output_dict=True,zero_division=0)
        for class_i in range(config.class_num):
            measure = report[str(class_i)]["precision"]
            class_precisionList_dict[class_i].append(measure)
    # 基于均值排名(均值越低排名越靠前)
    priority_queue = queue.PriorityQueue()
    for class_i in range(config.class_num):
        precision_list = class_precisionList_dict[class_i]
        avg_precision = round(sum(precision_list)/len(precision_list),4)
        item = (avg_precision,class_i)
        priority_queue.put(item)
    # 获得类别排名
    priority_list = priorityQueue_2_list(priority_queue)
    classes_rank = [class_i for priority,class_i in priority_list]
    return classes_rank

def detect_by_precision_and_loss():
    '''
    通过precision和loss排名进行怀疑集检测
    '''
    pass

'''
========普通功能函数区============
'''




'''
=======数据加载区=========
'''
def load_df(rate,df_name:str):
    if df_name == "preLabel":
        df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate),
                "preLabel.csv")
        )
    elif df_name == "CELoss":
        df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate),
                "CELoss.csv")
        )
    return df

'''
========数据保存区============
'''

'''
==========结果展示区==============
'''


'''
=======主函数区======
'''
def main(method:str):
    threshold = 0.2 # 25%的类别作为怀疑类别
    for rate in config.fine_mutation_rate_list:
        if method == "detect_by_precision":
            df = load_df(rate,"preLabel")
            class_rank = detect_by_precision(df,mutated_model_topk=50)
            cut_off = int(config.class_num*threshold)
            suspected_classes = class_rank[:cut_off]
            exp_logger.debug(f"rate:{rate}")
            exp_logger.debug(f"class_rank:{class_rank}")
            exp_logger.debug(f"suspected_classes:{suspected_classes}")
            exp_logger.debug("="*30)
        if method == "detect_by_loss":
            df = load_df(rate,"CELoss")
            class_rank = detect_by_loss(df,mutated_model_topk=50)
            cut_off = int(config.class_num*threshold)
            suspected_classes = class_rank[:cut_off]
            exp_logger.debug(f"rate:{rate}")
            exp_logger.debug(f"class_rank:{class_rank}")
            exp_logger.debug(f"suspected_classes:{suspected_classes}")
            exp_logger.debug("="*30)


if __name__ == "__main__":
    # 全局变量区
    dataset_name = config.dataset_name
    model_name = config.model_name
    attack_name = config.attack_name
    exp_time = get_formattedDateTime()
    # 实验元信息
    exp_info = {
        "exp_obj":"|".join([dataset_name,model_name,attack_name]),
        "exp_name":"Detect_Suspected_Classes",
        "exp_time":exp_time,
        "exp_method":"percent25",
        "args":{
            "measure_name":"precision", # default:"precision"
            "TopK":50,
            "diff_trend":"bigger" # type:str,default:smaller,condidate val:dsmaller|bigger|None
        }
    }
    # logg配置
    # 获得当前实验的日志记录者
    exp_log_file_dir = os.path.join("log",dataset_name,model_name,attack_name)
    exp_log_file_name = "_".join([exp_info["exp_name"],exp_time,".log"])
    exp_logger = get_Logger("BottomLevel",exp_log_file_dir,exp_log_file_name,filemode="w")
    exp_logger.debug("The exp info is:\n%s",json.dumps(exp_info,indent=4))

    # 获得实验流程的日志记录者
    record_logger = get_Logger("HighLevel",log_file_dir="log",log_file_name="exp_process_record.log",filemode="a")
    record_logger.debug(exp_info)
    record_logger.debug(f"exp_log in:{os.path.join(exp_log_file_dir,exp_log_file_name)}")

    # 实验开始
    main(method="detect_by_precision")



