'''
累计图
'''
import os
import pandas as pd
from codes import config
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from codes.ourMethod.draw.common import draw_line


def get_mutated_models_acc_list(df):
    # 得到每个变异模型的acc
    m_models_acc_list = []
    gt_label_list = df["GT_label"]
    for model_i in range(500):
        col_name = f"model_{model_i}"
        pre_label_list = list(df[col_name])
        report = classification_report(gt_label_list,pre_label_list,output_dict=True,zero_division=0)
        m_models_acc_list.append(report["accuracy"])
    return m_models_acc_list

def get_data(csv_path,acc_threshold_list):
    # 加载csv
    df = pd.read_csv(csv_path)
    m_models_acc_list = get_mutated_models_acc_list(df)
    count_list = []
    for threshold in acc_threshold_list:
        count = 0
        for m_acc in m_models_acc_list:
            if m_acc < threshold:
                count+=1
        count_list.append(count)
    return count_list

if __name__ == "__main__":
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    # 小于的区间
    acc_threshold_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # 存每个mutated rate的数据
    data_dict = {}
    for rate in config.fine_mutation_rate_list:
        # 该变异率下的变异模型集的预测标签结果csv文件
        csv_path = os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate),
                "preLabel.csv"
            )
        # 变异率下的变异模型集的acc累计统计结果
        count_list = get_data(csv_path,acc_threshold_list)
        # 记录到字典中
        data_dict[rate] = count_list
    # 绘制曲线
    x_ticks = acc_threshold_list
    title = "cumulative"
    xlabel = "acc"
    ylabel = "mutated model nums"
    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name,
    )
    save_file_name = "cumulative.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_line(x_ticks, title, xlabel, ylabel, save_path, data_dict)
    print(f"save_path:{save_path}")

    
    
    

