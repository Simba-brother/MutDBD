import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import statistics

from codes import config
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids

def get_class_dict(df,selected_model_i_list=list(range(500))):
    '''
    基于df,获得变异模型在每个分类的上的平均CE_loss.
    return:
        {
            class_idx:avg_ce_loss
        }
    '''
    data_dict = {}
    # 变异模型全局id列表:[0,1,2,..,400]
    m_i_list = selected_model_i_list
    # 遍历每个分类，从df中抽取分类i的df
    for class_i in range(config.class_num):
        df_class_i = df.loc[df["GT_label"]==class_i]
        # 保存该类别(class_i)的平均ce loss
        CE_loss_list = []
        # 遍历每个变异模型
        for m_i in m_i_list:
            # 计算该模型在该类别上的ce_loss
            ce_loss_list = df_class_i[f"model_{m_i}"]
            CE_loss_list.append(round(sum(ce_loss_list)/len(ce_loss_list),4))
        avg = statistics.mean(CE_loss_list)
        median = statistics.median(CE_loss_list)
        var = statistics.variance(CE_loss_list)
        data_dict[class_i] = avg
    return data_dict

def get_rate_class_dict():
    data_dict = {}
    for rate in config.fine_mutation_rate_list:
        CELoss_data_path = os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "CELoss.csv"
        )
        preLabel_data_path = os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "preLabel.csv"
        )
        preLabel_df = pd.read_csv(preLabel_data_path)
        selected_model_i_list = get_top_k_global_ids(preLabel_df,top_k=50,trend="bigger")
        df = pd.read_csv(CELoss_data_path)
        class_dict = get_class_dict(df,selected_model_i_list)
        data_dict[rate] = class_dict
    return data_dict


def draw_line(x_ticks:list, title:str, xlabel:str, ylabel:str, save_path:str, draw_data_dict:dict):
    # 设置图片大小，清晰度
    # plt.figure(figsize=(20, 8), dpi=800)
    x_list = [x for x in list(range(len(x_ticks)))]
    for key,value in draw_data_dict.items():
        if key == 3:
            plt.plot(x_list, value, label=key, marker='o', color="black")
        else:
            plt.plot(x_list, value, label=key, marker='o')
    # 设置x轴的刻度
    font_size=10
    plt.xticks(x_list,x_ticks,fontsize=font_size) # rotation=45
    plt.xlabel(xlabel,fontsize=font_size)
    plt.ylabel(ylabel,fontsize=font_size)
    plt.title(title,fontsize=font_size)
    plt.tight_layout(pad=0)
    # 绘制网格(控制透明度)
    plt.grid(alpha=0.5, linestyle=':')
    # 添加图例
    plt.legend(ncol=5)
    plt.show()
    plt.savefig(save_path,transparent=False,dpi=600)


def main():
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    # 得到各个变异率下各个分类的avg_ce_loss
    '''
    {
        rate:{
            class_i:avg_loss
        }
    }
    '''
    rate_class_dict = get_rate_class_dict()

    # 存各个分类下在各个变异率下的avg_ce_loss
    '''
    {
        class_i:[loss_0.01,loss_0.03,...,loss_0.1]
    }
    '''
    data_dict = defaultdict(list)
    for class_i in range(config.class_num):
        for rate in config.fine_mutation_rate_list:
            data_dict[class_i].append(rate_class_dict[rate][class_i])
    # 绘制折线图
    x_ticks = [0.01,0.03,0.05,0.07,0.09,0.1]
    title = "The loss of each category changes with the mutation rate"
    xlabel = "Mutation rate"
    ylabel = "Loss"
    save_path = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        "Loss.png"
    )
    draw_line(x_ticks, title, xlabel, ylabel, save_path, data_dict)
    print(f"save_path:{save_path}")
if __name__ == "__main__":
    main()