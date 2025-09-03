

import os
import matplotlib as mpl
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def draw_box(origin_data,save_path):
    '''origin_data:{m_rate:list}'''
    data = []
    for m_rate in origin_data.keys(): # [0.01,0.03,0.05,0.07,0.09,0.1]
        data.append(origin_data[m_rate])

    # 设置IEEE/Science风格的绘图参数
    plt.style.use(['science','ieee'])
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['mathtext.fontset'] = 'stix'


    labels = ['1%', '3%', '5%', '7%', '9%', '10%']  # 横轴labels

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
    plt.savefig(save_path)


def draw_cutoff(CutOff_list,PN_rate_list,ASR_list,ACC_list, save_path):
    '''论文配图，讨论截取阈值对PN/ASR/ACC的影响'''
    # 设置Science期刊风格
    # plt.style.use(['science','ieee'])
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 数据
    thresholds = CutOff_list
    PN_rate = PN_rate_list
    ASR = ASR_list
    ACC = ACC_list

    # 创建图形
    fig, ax = plt.subplots(figsize=(4.0, 2.5))  # IEEE双栏标准宽度3.5英寸
    ax.grid(True, linestyle=':', alpha=0.7)

    # 绘制三条折线
    line1, = ax.plot(thresholds, PN_rate, 'o-', color='#1f77b4', linewidth=1.2, markersize=4, 
                    markeredgecolor='w', markeredgewidth=0.3, label='PN Rate')
    line2, = ax.plot(thresholds, ASR, 's--', color='#d62728', linewidth=1.2, markersize=4, 
                    markeredgecolor='w', markeredgewidth=0.3, label='ASR')
    line3, = ax.plot(thresholds, ACC, '^-.', color='#2ca02c', linewidth=1.2, markersize=4, 
                    markeredgecolor='w', markeredgewidth=0.3, label='ACC')

    # 设置坐标轴
    ax.set_xlabel('Cutoff Threshold', fontsize=12)
    # ax.set_ylabel('Rate / Accuracy', fontsize=12)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{int(t*100)}%' for t in thresholds],rotation=-45)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 将Y轴显示为百分比

    # 添加图例 - 调整位置和样式
    ax.legend(loc='best', frameon=True, framealpha=0.8, fontsize=7, handlelength=2)
    # 添加网格 - 使用更细的网格线
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)


    # 设置底部和左侧边框为更细的线
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    # 添加标题
    # ax.set_title('Performance Metrics vs. Threshold', fontsize=14, pad=15)

    # 紧凑布局
    plt.tight_layout(pad=0.5)

    # 添加数据点标签（可选，根据需要取消注释）
    # for i, (x, y) in enumerate(zip(thresholds, PN_rate)):
    #     ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
    #                 xytext=(0,10), ha='center', fontsize=8)
    # for i, (x, y) in enumerate(zip(thresholds, ASR)):
    #     ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
    #                 xytext=(0,-15), ha='center', fontsize=8)
    # for i, (x, y) in enumerate(zip(thresholds, ACC)):
    #     ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
    #                 xytext=(0,10), ha='center', fontsize=8)

    # 保存图像（支持多种格式）
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # plt.savefig(os.path.join(save_dir,f"{scence_name}.pdf"), dpi=300, bbox_inches='tight')