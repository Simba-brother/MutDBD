'''分析不同变异率对class rank的影响'''
import os
import joblib
from utils.common_utils import read_yaml,get_class_num
import matplotlib as mpl
import scienceplots
import matplotlib.pyplot as plt
from collections import defaultdict

config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]

def one_scene(dataset_name,model_name,attack_name,metric="FP"): 
    mutation_rate_list = [0.01,0.03,0.05,0.07,0.09,0.1]
    for mutation_rate in mutation_rate_list:
        res = joblib.load(os.path.join(exp_root_dir,"Exp_Results","ClassRank",dataset_name,model_name,attack_name,str(mutation_rate),f"{metric}.joblib"))
        res["target_class_rank_ratio"]
        print(f"mutation_rate:{mutation_rate},rank_rate:{res["target_class_rank_ratio"]}")

def draw_box():
    '''论文插图：不同变异率对FP class rank的影响'''
    # 加载数据
    read_data = joblib.load(os.path.join(exp_root_dir,"Exp_Results","disscution_mutation_rate_for_class_rank.pkl"))
    # {mutation_rate:list}
    conver_data = defaultdict(list)
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
            class_num = get_class_num(dataset_name)
            for model_name in ["ResNet18","VGG19","DenseNet"]:
                for attack_name in ["BadNets","IAD","Refool","WaNet"]:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    conver_data[m_rate].append(read_data[dataset_name][model_name][attack_name][m_rate])
    data = []
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        data.append(conver_data[m_rate])
    # 设置IEEE/Science风格的绘图参数
    plt.style.use(['science','ieee'])
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'stix'


    labels = ['1%', '3%', '5%', '7%', '9%', '10%']

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制箱线图
    boxplot = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    # 自定义箱线图外观
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.7, linewidth=1.5)
        
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linewidth=1.5, linestyle='--')
        
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=1.5)
        
    for median in boxplot['medians']:
        median.set(color='red', linewidth=2)
        
    for mean in boxplot['means']:
        mean.set(marker='o', markerfacecolor='green', markeredgecolor='green', markersize=8)

    # # 计算并标注中位值和均值
    # for i, d in enumerate(data):
    #     median = np.median(d)
    #     mean = np.mean(d)
    #     # 标注中位数
    #     ax.text(i+1, median, f'{median:.3f}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    #     # 标注均值
    #     ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='top', fontsize=9, color='green', fontweight='bold')

    # 添加标签和标题
    ax.set_xlabel('Mutation Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank Ratio', fontsize=14, fontweight='bold')
    # ax.set_title('Rank Ratio Distribution at Different Mutation Rates', fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    # plt.show()
    plt.savefig("imgs/1.png")


if __name__ == "__main__":
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    one_scene(dataset_name,model_name,attack_name)
