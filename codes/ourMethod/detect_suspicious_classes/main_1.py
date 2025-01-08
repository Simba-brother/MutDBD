import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from codes import config
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids
from codes.common.time_handler import get_formattedDateTime
from codes.common.logging_handler import get_Logger

def measure_by_model_precision(df:pd.DataFrame,mutated_model_global_id_list:list):
    data_dict = defaultdict(list)
    # ground truth label
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(config.class_num):
            measure = report[str(class_i)]["precision"]
            data_dict[class_i].append(measure)
    return data_dict

def get_class_measure_avg(data_dict):
    ans_dict = {}
    for class_i in range(config.class_num):
        avg = round(sum(data_dict[class_i])/len(data_dict[class_i]),4)
        ans_dict[class_i] = avg
    return ans_dict

def main():
    # 加载preLabel_csv
    my_dict = defaultdict(list)
    for rate in config.fine_mutation_rate_list: # 从小到大
        df = pd.read_csv(os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "preLabel.csv"))
        mutated_model_global_id_list = get_top_k_global_ids(df,top_k=50,trend="bigger")
        measure_dict = measure_by_model_precision(df,mutated_model_global_id_list)
        avg_dict = get_class_measure_avg(measure_dict)
        for class_i in range(config.class_num):
            my_dict[class_i].append(avg_dict[class_i])

    speed_dict = {}
    for class_i in range(config.class_num):
        speed_dict[class_i] = round((my_dict[class_i][0]-my_dict[class_i][-1])/5,4)

    # [(class_i,speed)]
    items = list(speed_dict.items())
    #[(class_x,speed)]
    items.sort(key=lambda x:x[1],reverse=False) # 从小到大,下降速度越慢排名越高
    # [class_x,class_y]
    key_list = [key for key in items.keys()]
    look_data = {}
    for i in range(key_list):
        key = key_list[i]
        look_data[key] = i
    print(look_data)


if __name__ == "__main__":
    exp_time = get_formattedDateTime()
    # 实验元信息
    exp_info = {
        "exp_obj":"|".join([config.dataset_name,config.model_name,config.attack_name]),
        "exp_name":"Detect_Suspected_Classes",
        "exp_time":exp_time,
        "exp_method":"precision_drop_speed",
    }
        # 获得当前实验的日志记录者
    exp_log_file_dir = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    exp_log_file_name = "_".join([exp_info["exp_name"],exp_time,".log"])
    exp_logger = get_Logger("BottomLevel",exp_log_file_dir,exp_log_file_name,filemode="w")
    exp_logger.debug(exp_info)

    # 获得实验流程的日志记录者
    record_logger = get_Logger("HighLevel",log_file_dir="log",log_file_name="exp_process_record.log",filemode="a")
    record_logger.debug(exp_info)
    record_logger.debug(f"exp_log in:{os.path.join(exp_log_file_dir,exp_log_file_name)}")
    main()