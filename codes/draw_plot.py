'''
绘制论文中的数据图
'''
import os # 用于文件路径
import joblib # 用于加载数据
import scienceplots # 科学风格绘图
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_classNum(dataset_name):
    class_num = None
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    return class_num

def motivation_FPs():
    '''
    论文配图，动机章节：FP柱状图
    '''
    # 加载数据
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "WaNet"
    class_nums = 10 # 数据集类别数量
    selected_mutation_model_num = 50 # 选择的变异模型数量
    data = joblib.load(os.path.join(exp_root_dir,
                                    "实验结果",
                                    "标签迁移",
                                    "变异率0.01",
                                    dataset_name,model_name,attack_name,
                                    "res.joblib"))
    
    categories = list(range(class_nums))
    bar_data = []
    for c_i in categories:
        bar_data.append(data[c_i]/selected_mutation_model_num)

    # 应用IEEE风格
    plt.style.use(['science', 'ieee'])

    # 创建图形和坐标轴 - 使用IEEE推荐尺寸
    fig, ax = plt.subplots(figsize=(3.5, 2.2))  # IEEE双栏标准宽度3.5英寸

    # 绘制柱状图
    bar_colors = ['green' if i != 3 else 'red' for i in range(len(categories))]
    bars = ax.bar(categories, bar_data, color=bar_colors, 
                edgecolor='black', linewidth=0.5, alpha=0.8)

    # 设置x轴刻度和标签
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)

    # 计算Y轴上限，为最高柱子留出足够空间
    max_value = max(bar_data)
    y_margin = max_value * 0.1  # 10%的边距
    ax.set_ylim(0, max_value + y_margin)

    # 添加数值标签，确保不会与上边界重叠
    for bar in bars:
        height = bar.get_height()

        va = 'bottom'
        y_text = 2  # 正偏移，将标签放在柱子外部顶部
        color = 'black'
        
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_text),
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=6,
                    color=color)  # 设置文本颜色


    # 设置坐标轴标签
    ax.set_xlabel('Class')
    ax.set_yticks(np.linspace(0, max_value + y_margin, 5))  # 设置合理的Y轴刻度
    ax.set_ylabel('FPs')

    # 添加网格线（IEEE风格通常使用更细的网格）
    ax.grid(True, which='major', axis='y', linestyle=':', linewidth=0.5)

    # 紧凑布局
    plt.tight_layout(pad=0.5)

    # 保存图像（符合IEEE投稿要求）
    plt.savefig(f"imgs/Motivation/FPs/{attack_name}.png", dpi=600, bbox_inches='tight')

def discussion_mutation_rate():
    # 加载数据
    read_data = joblib.load(os.path.join(exp_root_dir,"实验结果","disscution_mutation_rate_for_class_rank.pkl"))
    # {mutation_rate:list}
    conver_data = defaultdict(list)
    for m_rate in [0.01,0.03,0.05,0.07,0.09,0.1]:
        for dataset_name in ["CIFAR10","GTSRB","ImageNet2012_subset"]:
            class_num = get_classNum(dataset_name)
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
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'


    labels = ['1\%', '3\%', '5\%', '7\%', '9\%', '10\%'] # IEEE样式默认使用LaTeX，

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(3.5, 2.5)) # IEEE双栏标准宽度3.5英寸

    # 绘制箱线图
    boxplot = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True, widths=0.6)

    # 自定义箱线图外观
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.7, linewidth=0.8)
        
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linewidth=0.8, linestyle='--')
        
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=0.8)
        
    for median in boxplot['medians']:
        median.set(color='red', linewidth=1.2)
        
    for mean in boxplot['means']:
        mean.set(marker='o', markerfacecolor='green', markeredgecolor='green', markersize=5)

    # # 计算并标注中位值和均值
    # for i, d in enumerate(data):
    #     median = np.median(d)
    #     mean = np.mean(d)
    #     # 标注中位数
    #     ax.text(i+1, median, f'{median:.3f}', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    #     # 标注均值
    #     ax.text(i+1, mean, f'{mean:.3f}', ha='center', va='top', fontsize=9, color='green', fontweight='bold')

    # 添加标签和标题
    # 添加标签
    ax.set_xlabel('Mutation Rate')
    ax.set_ylabel('Rank Ratio')


    # 设置底部和左侧边框为更细的线
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    # ax.set_xlabel('Mutation Rate', fontsize=14, fontweight='bold')
    # ax.set_ylabel('Rank Ratio', fontsize=14, fontweight='bold')
    # ax.set_title('Rank Ratio Distribution at Different Mutation Rates', fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    # plt.show()
    # 保存图像（符合IEEE投稿要求）
    plt.savefig("imgs/discussion/mutation_rate.png", dpi=600, bbox_inches='tight')




if __name__  == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments" # 项目实验根目录

    # motivation_FPs()
    discussion_mutation_rate()