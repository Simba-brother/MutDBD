'''
从上帝视角分析poisoned_trainset和clean_trainset在变异模型集上的差异。
比如confidence,accuracy,recall,precision,f1,LCR(标签变化率)等
'''
import os
import torch
import logging
import setproctitle
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import matplotlib.pyplot as plt

from codes import config
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
from codes.bigUtils import entropy



def cal_LCR(df:pd.DataFrame):
    # 首先计算original_backdoor_model预测的标签
    prob_outputs_list = list(df["original_backdoor_model"])
    original_label_list = []
    for prob_outpus in prob_outputs_list:
        # 得到top1 confi
        max_confi = max(prob_outpus)
        # 得到label index
        label_index = prob_outpus.index(max_confi)
        original_label_list.append(label_index)
    df["original_label"] = original_label_list
    # 计算每个样本的标签变化率
    LCR_list = []
    for _,row in df.iterrows():
        LC_count = 0
        total_count = 0
        label_o = row["original_label"]
        for model_global_id in range(500):
            col_name = f"model_{model_global_id}"
            label = row[col_name]
            if label != label_o:
                LC_count += 1
            total_count += 1
        LC_Rate = LC_count/total_count
        LCR_list.append(LC_Rate)
    return LCR_list

def cal_Entropy(df:pd.DataFrame):
    # 计算每个样本的entropy
    entropy_list = []
    for _,row in df.iterrows():
        label_list = []
        for model_global_id in range(500):
            col_name = f"model_{model_global_id}"
            label = row[col_name]
            label_list.append(label)
        entropy_list.append(entropy(label_list))
    return entropy_list

def cal_accuracy(df):
    # 首先计算original_backdoor_model预测的标签
    prob_outputs_list = list(df["original_backdoor_model"])
    original_label_list = []
    for prob_outpus in prob_outputs_list:
        # 得到top1 confi
        max_confi = max(prob_outpus)
        # 得到label index
        label_index = prob_outpus.index(max_confi)
        original_label_list.append(label_index)

    GT_labels = df["GT_label"]
    report = classification_report(GT_labels,original_label_list,output_dict=True,zero_division=0)
    acc_o = report["accuracy"]
    acc_m_list = []

    for model_global_id in range(500):
        col_name = f"model_{model_global_id}"
        pred_labels = list(df[col_name])
        report = classification_report(GT_labels,pred_labels,output_dict=True,zero_division=0)
        acc_m_list.append(report["accuracy"])
    
    acc_o_acc_m_absDif_list = []
    for acc_m in acc_m_list:
        absDif = abs(acc_o - acc_m)
        acc_o_acc_m_absDif_list.append(absDif)

    return acc_m_list,acc_o_acc_m_absDif_list

def draw_box(data_dict,save_path):
    
    
    clean_data = [data_dict["LCR"]["Clean"], data_dict["Entropy"]["Clean"], data_dict["ACC"]["Clean"],data_dict["ACC_dif"]["Clean"]]
    poisoned_data = [data_dict["LCR"]["Poisoned"], data_dict["Entropy"]["Poisoned"], data_dict["ACC"]["Poisoned"],data_dict["ACC_dif"]["Poisoned"]]
    # 设置箱线图宽度
    width = 0.2

    # 绘制箱线图
    plt.figure(figsize=(8, 6))

    xticks = [1,3,5,7]
    xticks_label = ["LCR","Entropy","Acc","AccDif"]
    # 绘制每个横坐标上的箱线图
    for i, (clean, poisoned) in enumerate(zip(clean_data, poisoned_data)):
        # 绘制 Clean 组
        plt.boxplot(clean, positions=[xticks[i] - width], widths=width, patch_artist=True, boxprops=dict(facecolor="skyblue", color="blue"), medianprops=dict(color="black"),showmeans=True)
        
        # 绘制 Poisoned 组
        plt.boxplot(poisoned, positions=[xticks[i] + width], widths=width, patch_artist=True, boxprops=dict(facecolor="salmon", color="red"), medianprops=dict(color="black"),showmeans=True)

    # 设置横轴标签
    # 设置横坐标刻度（控制刻度的范围和步长）
    plt.xticks(xticks,xticks_label)
    # 旋转横坐标标签，避免重叠
    plt.xticks(rotation=45)
    # 设置轴标签
    plt.xlabel('indicator')
    plt.ylabel('performance')

    # 设置图例
    plt.legend(["Clean","Poisoned"])

    # 添加标题
    plt.title('Boxplot for Clean and Poisoned Data')

    # 显示图形
    plt.tight_layout(pad=0)
    # 绘制网格(控制透明度)
    plt.grid(alpha=0.5)
    plt.savefig(save_path,transparent=False,dpi=600)



def main(mutation_rate):
    logging.debug(f"变异率:{mutation_rate}")
    csv_df = pd.read_csv(os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(mutation_rate),
        "preLabel.csv"))
    csv_df["original_backdoor_model"] = prob_outputs
    # 得到2组数据
    df_poisoned = csv_df.loc[csv_df["isPoisoned"]==True]
    df_clean = csv_df.loc[csv_df["isPoisoned"]==False]

    ans = {}
    # 计算LCR（样本角度）
    poisoned_LCR_list = cal_LCR(df_poisoned)
    clean_LCR_list = cal_LCR(df_clean)

    # 计算Entropy
    poisoned_Entropy_list = cal_Entropy(df_poisoned)
    clean_Entropy_list = cal_Entropy(df_clean)

    # 计算Accuracy and |ACC_o-ACC_m|（模型角度）
    poisoned_ACC_list,poisoned_absACCdif_list = cal_accuracy(df_poisoned)
    clean_ACC_list,clean_absACCdif_list = cal_accuracy(df_clean)

    ans["LCR"] = {
        "Poisoned":poisoned_LCR_list,
        "Clean":clean_LCR_list,
        
    }

    ans["Entropy"] = {
        "Poisoned":poisoned_Entropy_list,
        "Clean":clean_Entropy_list,
    }

    ans["ACC"] = {
        "Poisoned":poisoned_ACC_list,
        "Clean":clean_ACC_list
    }

    ans["ACC_dif"] = {
        "Poisoned":poisoned_absACCdif_list,
        "Clean":clean_absACCdif_list
    }
    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(mutation_rate))
    os.makedirs(save_dir,exist_ok=True)
    file_name = "Analysis_of_Differential_Indicators.png"
    save_path = os.path.join(save_dir,file_name)
    draw_box(ans,save_path)
    logging.debug(f"save_path:{save_path}")


if __name__ == "__main__":
    # 进程名称
    proctitle = f"Analysis_of_Differential_Indicators|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cuda:{config.gpu_id}")

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log",config.dataset_name,config.model_name,config.attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = "Analysis_of_Differential_Indicators.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)
    try:
        # 加载后门数据
        backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        # 原始的后门模型
        backdoor_model = backdoor_data["backdoor_model"]
        poisoned_trainset =backdoor_data["poisoned_trainset"]
        # 评估原始后门模型在poisoned_trainset上的指标
        device = torch.device(f"cuda:{config.gpu_id}")
        e = EvalModel(backdoor_model,poisoned_trainset,device)
        prob_outputs = e.get_prob_outputs()
        for mutation_rate in config.fine_mutation_rate_list:
            main(mutation_rate)
    except Exception as e:
        logging.debug("发生异常:%s",e)