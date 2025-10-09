'''
从上帝视角分析poisoned_trainset和clean_trainset在变异模型集上的差异。
比如confidence,accuracy,recall,precision,f1,LCR(标签变化率)等
'''
import os
import torch
import logging
import setproctitle
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
import matplotlib
import scienceplots
import matplotlib.pyplot as plt
import joblib
import numpy as np

from codes import config
from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
from codes.utils import entropy
from codes.ourMethod.detect_suspicious_classes.select_mutated_model import get_top_k_global_ids



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



def get_mutation_acc_list(df,mutated_model_id_list):
    GT_labels = df["GT_label"]
    if mutated_model_id_list is None:
        mutated_model_id_list = list(range(500))
    mutation_acc_list = []
    for model_global_id in mutated_model_id_list:
        col_name = f"model_{model_global_id}"
        pred_labels = list(df[col_name])
        report = classification_report(GT_labels,pred_labels,output_dict=True,zero_division=0)
        mutation_acc_list.append(report["accuracy"])
    return mutation_acc_list

def main_2():

    csv_df = pd.read_csv(os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        "CIFAR10",
        "ResNet18",
        "BadNets",
        str(0.01),
        "preLabel.csv"))
    grid = joblib.load(os.path.join(config.exp_root_dir,"Exp_Results","grid.joblib"))
    mutated_model_id_list = grid["CIFAR10"]["ResNet18"]["BadNets"][0.01]["top50_model_id_list"]

    df_poisoned = csv_df.loc[(csv_df["isPoisoned"]==True) & (csv_df["GT_label"]==3)]
    df_clean = csv_df.loc[(csv_df["isPoisoned"]==False) & (csv_df["GT_label"]==3)]
    poisoned_acc_list = get_mutation_acc_list(df_poisoned,mutated_model_id_list)
    clean_acc_list = get_mutation_acc_list(df_clean,mutated_model_id_list)
    res = {
        "poisoned_acc_list":poisoned_acc_list,
        "clean_acc_list":clean_acc_list
    }
    save_dir = os.path.join(config.exp_root_dir, "Exp_Results","中毒干净acc分布","变异率0.01", "CIFAR10","ResNet18","BadNets")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(res,save_path)
    print(save_path)
    print("END")


def main_3():
    dataset_name = "CIFAR10"  # ImageNet2012_subset
    model_name = "ResNet18"
    attack_name = "WaNet"
    csv_df = pd.read_csv(os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        dataset_name,
        model_name,
        attack_name,
        str(0.01),
        "preLabel.csv"))
    
    mutated_model_id_list = get_top_k_global_ids(csv_df)
    # grid = joblib.load(os.path.join(config.exp_root_dir,"Exp_Results","grid.joblib"))
    # mutated_model_id_list = grid["CIFAR10"]["ResNet18"]["IAD"][0.01]["top50_model_id_list"]
    
    # grid = joblib.load(os.path.join(config.exp_root_dir,"Exp_Results","ImageNet_grid.joblib"))
    # mutated_model_id_list = grid[dataset_name][model_name][attack_name][0.01]["top50_model_id_list"]
    
    '''
    res = {}
    for m_i in mutated_model_id_list:
        res[m_i] = {}
        pre_labels = csv_df[f"model_{m_i}"]
        gt_labels = csv_df[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(10):
            res[m_i][class_i] = {}
            error_noTarget_num = 0
            error_target_num = 0
            correct_num = cm[class_i][class_i]
            for o_i in range(10):
                if o_i != class_i and o_i != 3:
                    error_noTarget_num += cm[class_i][o_i]
                if o_i != class_i and o_i == 3:
                    error_target_num += cm[class_i][o_i]
            res[m_i][class_i]["correct_num"] = correct_num 
            res[m_i][class_i]["error_noTarget_num"] = error_noTarget_num 
            res[m_i][class_i]["error_target_num"] = error_target_num

    res_1 = {}
    for class_i in range(10):
        res_1[class_i] = {}
        correct_num = 0
        error_noTarget_num = 0
        error_target_num = 0
        for m_i in mutated_model_id_list:
            correct_num += res[m_i][class_i]["correct_num"]
            error_noTarget_num += res[m_i][class_i]["error_noTarget_num"]
            error_target_num += res[m_i][class_i]["error_target_num"]
        res_1[class_i]["correct_num"] = correct_num
        res_1[class_i]["error_noTarget_num"] = error_noTarget_num
        res_1[class_i]["error_target_num"] = error_target_num
    '''

    '''
    res = {}
    for m_i in mutated_model_id_list:
        res[m_i] = {}
        pre_labels = csv_df[f"model_{m_i}"]
        gt_labels = csv_df[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(10):
            res[m_i][class_i] = {}
            for o_i in range(10):
                res[m_i][class_i][o_i] = cm[class_i][o_i]
    res_1 = {}
    for class_i in range(10):
        res_1[class_i] = {}
        for o_i in range(10):
            res_1[class_i][o_i] = 0
            for m_i in mutated_model_id_list:
                res_1[class_i][o_i] += res[m_i][class_i][o_i]

    save_dir = os.path.join(config.exp_root_dir, "Exp_Results","标签迁移","变异率0.01", "CIFAR10","ResNet18","BadNets")
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(res_1,save_path)
    print(save_path)
    print("END")
    '''

    res = {}
    for m_i in mutated_model_id_list:
        res[m_i] = {}
        pre_labels = csv_df[f"model_{m_i}"]
        gt_labels = csv_df[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(10):
            res[m_i][class_i] = np.sum(cm[:,class_i]) - cm[class_i][class_i]

    res_1 = {}
    for class_i in range(10):
        res_1[class_i] = 0
        for m_i in mutated_model_id_list:
            res_1[class_i] += res[m_i][class_i]
    sorted_res_1 = dict(sorted(res_1.items(), key=lambda x: x[1],reverse=True))
    print(sorted_res_1)
    save_dir = os.path.join(config.exp_root_dir, "Exp_Results","标签迁移","变异率0.01", dataset_name,model_name,attack_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res.joblib"
    save_path = os.path.join(save_dir,save_file_name)
    joblib.dump(res_1,save_path)
    print(save_path)
    print("END")


def data_visualization_stackbar():
    # 加载数据
    data = joblib.load(os.path.join(config.exp_root_dir,
                                    "Exp_Results",
                                    "标签迁移",
                                    "变异率0.01",
                                    "CIFAR10","ResNet18","BadNets",
                                    "res.joblib"))
    categories = list(range(10))
    num_stacks = 10  # 每个柱子的堆叠部分数量
    stack_data = [[] for _ in range(10)]
    for c_id in categories: 
        for o_id in categories:
            if c_id == o_id:
                stack_data[o_id].append(0)
            else:
                stack_data[o_id].append(data[c_id][o_id])
    stack_data = np.array(stack_data)
    # 创建颜色映射（使用tab10色图，可自定义）
    colors = plt.cm.tab10(np.linspace(0, 1, num_stacks))

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(14, 8))

    # 初始化底部位置（从0开始）
    bottoms = np.zeros(len(categories))

    # 绘制堆叠柱状图
    for i in range(num_stacks):
        ax.bar(categories, stack_data[i], bottom=bottoms, 
            color=colors[i], label=f'part {i+1}')
        bottoms += stack_data[i]  # 更新底部位置

    # 添加总量标签（每个柱子的顶部）
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.5, f'{int(total)}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置图表标题和标签
    ax.set_title('Data distribution of each category (10-part stacked bar chart)', fontsize=16, pad=20)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Total quantity', fontsize=12)

    # 添加图例（放在图表外部右侧）
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Stacking part')

    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间
    plt.grid(axis='y', alpha=0.3)

    plt.show()
    plt.savefig("imgs/stack.png")

def data_visualization_bar():
    '''
    论文配图，动机章节：FP柱状图
    '''
    # 加载数据
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    data = joblib.load(os.path.join(config.exp_root_dir,
                                    "Exp_Results",
                                    "标签迁移",
                                    "变异率0.01",
                                    dataset_name,model_name,attack_name,
                                    "res.joblib"))
    categories = list(range(10))
    bar_data = []
    for c_i in categories:
        bar_data.append(data[c_i]/50)


    


    # 设置IEEE/Science风格的绘图参数
    plt.style.use(['science','ieee'])
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 3)) # IEEE双栏推荐宽度3.5英寸，这里适当放宽

    # 创建颜色列表：第4个柱子(索引3)为红色，其余为绿色
    bar_colors = ['green' if i != 3 else 'red' for i in range(len(categories))]
    # 绘制柱状图
    x_pos = np.arange(len(categories))  # 创建0-9的位置索引

    # 绘制柱状图
    bars = ax.bar(categories, bar_data, color=bar_colors, edgecolor='black',linewidth=0.5, alpha=0.8)

    # 设置x轴刻度和标签 - 关键修改：确保所有标签都显示
    ax.set_xticks(x_pos)  # 设置所有10个位置都有刻度
    ax.set_xticklabels(categories)  # 设置所有10个位置的标签
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # 设置标题和标签
    # ax.set_title('', fontsize=14, pad=20)
    # 设置坐标轴标签（Science风格通常使用更正式的标签）
    ax.set_xlabel('Class', fontsize=9, labelpad=2)
    ax.set_ylabel('FPs', fontsize=9, labelpad=2)
    # 调整刻度参数
    ax.tick_params(axis='both', which='major', labelsize=8, pad=2)

    # 添加网格线（保持简洁风格）
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)

    # 移除上部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 保存图像（设置高DPI和边界）
    plt.savefig(f"imgs/FP_{attack_name}.png", 
                dpi=600, 
                bbox_inches='tight',
                pad_inches=0.05)

    # 调整布局
    # plt.tight_layout()

    # 显示图形
    # plt.show()

    plt.savefig(f"imgs/FP_{attack_name}.png")


def data_visualization_box():
    # 加载数据
    data = joblib.load(os.path.join(config.exp_root_dir,
                                    "Exp_Results",
                                    "中毒干净acc分布",
                                    "变异率0.01",
                                    "CIFAR10","ResNet18","BadNets",
                                    "res.joblib"))
    clean_acc_list = data["clean_acc_list"]
    poisoned_acc_list = data["poisoned_acc_list"]
    box_data = [clean_acc_list,poisoned_acc_list]
    labels = ["Clean","Poisioned"]
    plt.figure(figsize=(10, 6))  # 设置画布大小
    # 绘制箱线图
    box = plt.boxplot(box_data, 
                    patch_artist=True,  # 允许填充颜色
                    showmeans=False,     # 先不显示默认均值标记
                    labels=labels)
    # 隐藏中位线
    for element in ['medians']:
        for line in box[element]:
            line.set_visible(False)  # 将中位数设置为不可见

    # 自定义箱体颜色 - 类别3为红色，其他为绿色
    colors = ["green","red"]
    # 应用颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # 设置透明度
    # 设置箱线图线条样式
    # plt.setp(box['medians'], color='black', linewidth=2)  # 中位数线
    plt.setp(box['whiskers'], color='gray', linestyle='--') 
    plt.setp(box['caps'], color='gray', linewidth=2)

    # 计算并添加均值线
    means = [np.mean(d) for d in box_data]  # 计算每个类别的均值
    for i, mean in enumerate(means):
        # 在每个箱线图位置添加均值横线
        plt.hlines(mean, 
                xmin=i+0.8, 
                xmax=i+1.2, 
                colors='black', 
                linewidth=3,
                label='mean' if i == 0 else "")  # 仅添加一次图例

    # 添加标签和标题
    # plt.xlabel('', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线
    plt.ylim(0.7, 1.2)
    # 添加图例
    # plt.legend(loc="upper right")
    plt.legend()

    # 优化布局并显示
    plt.tight_layout()
    plt.show()
    plt.savefig("imgs/clean_poisoned.png")

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

    '''
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
    '''

    # main_3()
    data_visualization_bar()
    # data_visualization_stackbar()
    # data_visualization()