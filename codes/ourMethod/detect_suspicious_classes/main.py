'''
检测target classes
'''
import os
from codes import config
import numpy as np
from scipy import stats
from cliffs_delta import cliffs_delta
import joblib
from codes.ourMethod.detect_suspicious_classes.get_suspicious import get_suspicious_classes_by_ScottKnottESD
from collections import defaultdict
import pandas as pd
from collections import defaultdict
from codes.utils import entropy
from sklearn.metrics import classification_report,confusion_matrix
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids
from codes.common.logging_handler import get_Logger
from codes.common.time_handler import get_formattedDateTime


def calu_p_and_dela_value(list_1, list_2):
    p_value = stats.wilcoxon(list_1, list_2).pvalue
    list_1_sorted = sorted(list_1) # 原来list不改变
    list_2_sorted = sorted(list_2)
    delta,info = cliffs_delta(list_1_sorted, list_2_sorted)
    return p_value,delta


def get_suspicious_classes_by_Rule(data_dict):
    '''
    data_dict:{Int(class_idx):list(precision|recall|F1)}
    '''
    # Rule_1:Wilcoxon Signed Rank Test and Cliff's Delta
    rule_1_classes = set()
    for i in range(config.class_num):
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
            rule_1_classes.add(i)
    
    # Rule_2:均值最大类
    rule_2_classes = set()
    max_avg = -1
    max_avg_class_idx = -1
    for i in range(config.class_num):
        array = np.array(data_dict[i])
        # 当前类别i的指标均值
        avg_value = np.round(np.mean(array),decimals=4).item()
        if avg_value > max_avg:
            max_avg = avg_value
            max_avg_class_idx = i
    rule_2_classes.add(max_avg_class_idx)

    # Rule_3:中位值最大类
    rule_3_classes = set()
    max_mid = -1
    max_mid_class_idx = -1
    for i in range(config.class_num):
        array = np.array(data_dict[i])
        # 当前类别i的指标均值
        mid_value = np.round(np.median(array),decimals=4).item()
        if mid_value > max_mid:
            max_mid = mid_value
            max_mid_class_idx = i
    rule_3_classes.add(max_mid_class_idx)

    suspicious_classes = rule_1_classes | rule_2_classes | rule_3_classes
    return suspicious_classes

def reconstruct_data(report_dataset,measure_name):
    '''
    args:
        report_dataset:
            {
                ratio:{
                    operation:[report_classification]
                }
            }
        measure_name:str,precision|recall|f1-score
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
        for class_i in range(config.class_num):
            data[ratio][class_i] = []
            for op in config.mutation_name_list:
                for report in report_dataset[ratio][op]:
                    data[ratio][class_i].append(report[str(class_i)][measure_name])
    return data

def detect(report_dataset,measure_name):
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
    data = reconstruct_data(report_dataset,measure_name)
    
    box_data_save_dir = os.path.join(config.exp_root_dir,"SK",config.dataset_name,config.model_name,config.attack_name)
    
    for ratio in config.fine_mutation_rate_list:
        save_dir = os.path.join(box_data_save_dir,str(ratio))
        os.makedirs(save_dir,exist_ok=True)
        save_file_name = f"box_{measure_name}.csv"
        save_path = os.path.join(save_dir,save_file_name)
        df = pd.DataFrame(data[ratio])
        # 直接设置新的列名列表
        new_col_name_list = []
        for old_col_name in list(df.columns):
            new_col_name_list.append("C"+str(old_col_name))
        df.columns = new_col_name_list
        df.to_csv(save_path,index=False)
    for ratio in config.fine_mutation_rate_list:
        suspicious_classes= get_suspicious_classes_by_ScottKnottESD(data[ratio])
        ans[ratio] = suspicious_classes
    return ans

def main(measure_name,rate_list,isTopK,K,diff_trend):
    '''
    measure_name: precison|recall|f1-score|LCR|AccDif|confidence
    isTopK:是否取TopK的变异模型
    '''
    # 计算实验结果
    '''
    res = {rate:{"top":[],"low":[]}}
    '''
    res = {}
    exp_logger.debug("开始:获得每个变异率下的Suspicious Classes")
    for rate in rate_list: # config.fine_mutation_rate_list:
        exp_logger.debug(f"变异率:{rate}") 
        label_df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate),
                "preLabel.csv"))
        if measure_name == "confidence":
            confidence_df = pd.read_csv(os.path.join(
                    config.exp_root_dir,
                    "EvalMutationToCSV",
                    config.dataset_name,
                    config.model_name,
                    config.attack_name,
                    str(rate),
                    "confidence.csv"))
        if isTopK is False:
            mutated_model_global_id_list = list(range(500))
        else:
            mutated_model_global_id_list = get_top_k_global_ids(label_df,top_k=K,trend=diff_trend)
        if measure_name  == "precision":
            class_measure_dict = measure_by_model_precision(label_df,mutated_model_global_id_list)
        elif measure_name == "recall":
            class_measure_dict = measure_by_model_recall(label_df,mutated_model_global_id_list)
        elif measure_name == "f1-score":
            class_measure_dict = measure_by_model_f1_score(label_df,mutated_model_global_id_list)
        elif measure_name == "LCR":
            class_measure_dict = measure_by_model_LCR(label_df,mutated_model_global_id_list)
        elif measure_name == "AccDif":
            class_measure_dict = measure_by_model_AccDif(label_df,mutated_model_global_id_list)
        elif measure_name == "confidence":
            class_measure_dict = measure_by_model_confidence(confidence_df,mutated_model_global_id_list)
        '''
        # 把绘制箱线图的数据保存一下
        box_data_save_dir = os.path.join(config.exp_root_dir,"SK",config.dataset_name,config.model_name,config.attack_name)
        save_dir = os.path.join(box_data_save_dir,str(rate))
        os.makedirs(save_dir,exist_ok=True)
        save_file_name = f"box_{measure_name}.csv"
        save_path = os.path.join(save_dir,save_file_name)
        df = pd.DataFrame(class_measure_dict)
        # 直接设置新的列名列表
        new_col_name_list = []
        for old_col_name in list(df.columns):
            new_col_name_list.append("C"+str(old_col_name))
        df.columns = new_col_name_list
        df.to_csv(save_path,index=False)
        '''
        # SK计算
        suspicious_classes_top,suspicious_classes_low = get_suspicious_classes_by_ScottKnottESD(class_measure_dict)
        res[rate] = {"top":suspicious_classes_top,"low":suspicious_classes_low}
    # 日志记录实验数据
    exp_logger.debug(res)

    # 保存实验结果
    save_dir = os.path.join(
        config.exp_root_dir,
        "SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = f"SuspiciousClasses_SK_{measure_name}.data"
    save_file_path = os.path.join(save_dir,save_file_name)
    joblib.dump(res,save_file_path)
    exp_logger.debug(f"SuspiciousClasses结果保存在:{save_file_path}")

    # 规范化展示结果
    target_class_idx = config.target_class_idx
    for rate in rate_list:
        exp_logger.debug("="*10)
        exp_logger.debug(f"rate:{rate}")
        top_class_list = res[rate]["top"]
        low_class_list = res[rate]["low"]
        if target_class_idx in top_class_list:
            exp_logger.debug("Top_group")
            ranking_within_the_group = top_class_list.index(target_class_idx)
            exp_logger.debug(f"ranking_within_the_group:{ranking_within_the_group+1}/{len(top_class_list)}")
        if target_class_idx in low_class_list:
            exp_logger.debug("Low_group")
            ranking_within_the_group = low_class_list.index(target_class_idx)
            exp_logger.debug(f"ranking_within_the_group:{ranking_within_the_group+1}/{len(low_class_list)}")
        if (target_class_idx not in top_class_list) and (target_class_idx not in low_class_list):
            exp_logger.debug("Mid_group")
        exp_logger.debug("="*30)

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

def measure_by_model_recall(df:pd.DataFrame,mutated_model_global_id_list:list):
    data_dict = defaultdict(list)
    # ground truth label
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(config.class_num):
            measure = report[str(class_i)]["recall"]
            data_dict[class_i].append(measure)
    return data_dict

def measure_by_model_f1_score(df:pd.DataFrame,mutated_model_global_id_list:list):
    data_dict = defaultdict(list)
    # ground truth label
    GT_label_list = df["GT_label"]
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(config.class_num):
            measure = report[str(class_i)]["f1-score"]
            data_dict[class_i].append(measure)
    return data_dict

def measure_by_model_LCR(df:pd.DataFrame,mutated_model_global_id_list):
    data_dict = defaultdict(list)
    for class_id in range(config.class_num):
        class_df = df.loc[df["GT_label"]==class_id]
        original_model_pred_label_list = list(class_df["original_backdoorModel_preLabel"])
        for mutated_model_global_id in mutated_model_global_id_list:
            count = 0
            model_col_name = f"model_{mutated_model_global_id}"
            pred_label_list = list(class_df[model_col_name])
            for o_l,m_l in zip(original_model_pred_label_list,pred_label_list):
                if o_l != m_l:
                    count += 1
            lcr = round(count/len(original_model_pred_label_list),4)
            data_dict[class_id].append(lcr)
    return data_dict

def measure_by_model_AccDif(df:pd.DataFrame,mutated_model_global_id_list:list):
    '''
    注意：在单类别上acc === recall且precision === 1
    '''
    data_dict = defaultdict(list)
    # ground truth label
    GT_label_list = df["GT_label"]
    original_backdoorModel_preLabel_list = df["original_backdoorModel_preLabel"]
    report_o = classification_report(GT_label_list,original_backdoorModel_preLabel_list,output_dict=True, zero_division=0)
    for mutated_model_global_id in mutated_model_global_id_list:
        model_col_name = f"model_{mutated_model_global_id}"
        pred_label_list = list(df[model_col_name])
        report_m = classification_report(GT_label_list,pred_label_list,output_dict=True, zero_division=0)
        for class_i in range(config.class_num):
            measure = abs(report_m[str(class_i)]["recall"]-report_o[str(class_i)]["recall"])
            data_dict[class_i].append(measure)
    return data_dict

def measure_by_model_confidence(df:pd.DataFrame,mutated_model_global_id_list):
    data_dict = defaultdict(list)
    for class_id in range(config.class_num):
        class_df = df.loc[df["GT_label"]==class_id]
        for mutated_model_global_id in mutated_model_global_id_list:
            confidence_list = []
            model_col_name = f"model_{mutated_model_global_id}"
            confidence_list = list(class_df[model_col_name])
            avg_confidence = round(sum(confidence_list) / len(confidence_list),4)
            data_dict[class_id].append(avg_confidence)
    return data_dict

def measure_by_sample_entropy(df:pd.DataFrame,mutated_model_global_id_list:list):
    data_dict = defaultdict(list)
    for class_id in range(config.class_num):
        class_df = df.loc[df["GT_label"]==class_id]
        for row_id, row in class_df.iterrows():
            # 当前row(样本)在所有变异模型集上的预测标签
            cur_row_labels = []
            for mutated_model_global_id in mutated_model_global_id_list:
                model_col_name = f"model_{mutated_model_global_id}"
                cur_row_labels.append(row[model_col_name])
            # 计算该row(样本)的标签熵
            cur_entropy = entropy(cur_row_labels)
            data_dict[class_id].append(cur_entropy)
    return data_dict



if __name__ == "__main__":
    
    # 生成实验时间
    exp_time = get_formattedDateTime()
    # 构建实验元信息
    exp_info = {
        "exp_obj":"|".join([config.dataset_name,config.model_name,config.attack_name]),
        "exp_name":"Detect_Suspected_Classes", # 检测可疑类别集
        "exp_time":exp_time, # 实验时间
        "exp_method":"ScottKnottESD", # 实验使用的方法
        "args":{
            "measure_name":"precision", # default:"precision" 方法使用的度量
            "isTop":True,
            "K":50,  # type:int,default:50
            "diff_trend":"bigger" # type:str,default:smaller,condidate val:dsmaller|bigger|None
        }
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

    # 主程序
    try:
        # 主程序参数
        rate_list = config.fine_mutation_rate_list
        measure_name = exp_info["args"]["measure_name"]
        isTopK = exp_info["args"]["isTop"]
        K = exp_info["args"]["K"]
        diff_trend = exp_info["args"]["diff_trend"]
        # 主程序开始
        main(measure_name,rate_list,isTopK,K,diff_trend)
        exp_logger.debug("Successfully finish")
    except Exception as e:
        exp_logger.error("发生异常:%s",e)
        record_logger.error("发生异常:%s",e)
