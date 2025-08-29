
import os

import torch
from codes.models import get_model
from codes.datapoint import get_classes_rank_v2,get_backdoor_info
from codes.bigUtils.sample_split import _split
from codes.bigUtils.dataset import get_all_dataset
from codes.common.eval_model import EvalModel

# 绘图
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter

def main():
    # 加载两个空的模型结构
    select_model = get_model(dataset_name,model_name)
    defense_model = get_model(dataset_name,model_name)
    # 加载选择模型权重
    selected_state_dict_path = os.path.join(exp_root_dir,"cut_off",dataset_name,model_name,attack_name,f"exp_{random_seed}","best_BD_model.pth")
    select_model.load_state_dict(torch.load(selected_state_dict_path,"cpu"))

    class_rank_list = get_classes_rank_v2(exp_root_dir,dataset_name,model_name,attack_name)
    # 获得数据
    backdoor_data = get_backdoor_info(exp_root_dir,dataset_name,model_name,attack_name)
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name,model_name,attack_name,poisoned_ids)

    

    CutOff_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    PN_rate_list = []
    ASR_list = []
    ACC_list = []
    for cut_off in CutOff_list:
        # 加载防御模型权重
        defense_state_dict_path = os.path.join(exp_root_dir,"cut_off",dataset_name,model_name,attack_name,f"exp_{random_seed}",str(cut_off),"best_defense_model.pth")
        defense_model.load_state_dict(torch.load(defense_state_dict_path,"cpu"))
        # PN
        p_count, choiced_num, poisoning_rate = _split(select_model,poisoned_trainset,poisoned_ids,device,class_rank = class_rank_list,choice_rate = cut_off)
        # ASR
        em = EvalModel(defense_model, filtered_poisoned_testset, device)
        asr = em.eval_acc()
        em = EvalModel(defense_model,clean_testset, device)
        acc = em.eval_acc()

        PN_rate_list.append(round(p_count/len(poisoned_ids),4))
        ASR_list.append(asr)
        ACC_list.append(acc)
        print(f"cut_off完成:{cut_off}")

    return CutOff_list,PN_rate_list,ASR_list,ACC_list


def draw():
    '''论文配图，讨论截取阈值对PN/ASR/ACC的影响'''
    # 获得数据
    CutOff_list,PN_rate_list,ASR_list,ACC_list = main()
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

   
    save_dir = "imgs/discussion/cut_off"
    # 保存图像（支持多种格式）
    plt.savefig(os.path.join(save_dir,f"{attack_name}.png"), dpi=600, bbox_inches='tight')
    # plt.savefig(os.path.join(save_dir,f"{scence_name}.pdf"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":

    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "CIFAR10"
    model_name  = "ResNet18"
    attack_name = "WaNet"
    # scence_name = f"{dataset_name}_{model_name}_{attack_name}"
    random_seed = 1
    # 获得计算设备
    device = torch.device(f"cuda:0")
    draw()
    print("END")