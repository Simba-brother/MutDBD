
'''
使用每个类别的10个干净seed来辅助target class的确定
'''
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

import torch

from codes import config


def sampling_seed(id_list:list, poisoned_id_list:list, gt_label_list:list, num: int = 10) -> dict:
    '''
    从每个类别中采样出干净样本
    Args:
    ----
    id_list:list
        数据样本id_list
    poisoned_id_list:list
        中毒数据样本id_list
    gt_label_list:list
        数据样本真实分类标签
    num:int，defaut:10
        设置每个类别要采样的数量
    Return:
    ------
    '''
    # 存最后每个类别种子id_list结果
    seed_dict = defaultdict(list)
    # 存最后每个类别剩余id_list结果
    other_dict = defaultdict(list)
    # 求clean_id_list
    clean_id_list = list(set(id_list) - set(poisoned_id_list))
    # 按照class将clean_id_list分桶装
    class_dict = defaultdict(list)
    for clean_id,label in zip(clean_id_list,gt_label_list):
        class_dict[label].append(clean_id)
    
    for cls,id_list in class_dict.items():
        # 不放回采样
        seed_list = random.sample(id_list,num)
        seed_dict[cls].extend(seed_list)
        other_list = list(set(id_list) - set(seed_list))
        other_dict[cls].extend(other_list)
    return seed_dict,other_dict
    

def get_acc_dic(cls_dict):
    acc_dic = defaultdict(list)
    for cls in cls_dict.keys():
        ids = cls_dict[cls]
        sub_df = pre_label_df.loc[ids]
        for model_i in range(mutation_model_num):
            model_col = f"model_{model_i}"
            predict_label_list = sub_df[model_col].tolist()
            true_label_list = [cls]*len(predict_label_list)
            report = classification_report(true_label_list,predict_label_list, output_dict=True)
            # 当前模型model_i在cls类别上的干净种子的预测accuracy
            acc = report["accuracy"]
            acc_dic[cls].append(acc)
    return acc_dic
        
def main():
    # 随机抽取每个类中的干净seed
    seed_dict,other_dict = sampling_seed(id_list, poisoned_ids, gt_label_list, num=10)
    seed_acc_dic = get_acc_dic(seed_dict)
    other_acc_dict = get_acc_dic(other_dict)
    '''
    # 绘制各个类别上acc_list箱线图
    '''
    seeds = []
    others = [] 
    cls_list = sorted(seed_acc_dic.keys())
    for cls in cls_list:
        seed_acc_list = seed_acc_dic[cls]
        other_acc_list = other_acc_dict[cls]
        assert len(seed_acc_list) == mutation_model_num, "数量不对"
        assert len(other_acc_list) == mutation_model_num, "数量不对"
        seeds.append(seed_acc_list)
        others.append(other_acc_list)
    
    # 设置每个类别下两个箱线图的位置（左右偏移）
    categories = cls_list
    positions_seeds = [i - 0.2 for i in categories]
    positions_others = [i + 0.2 for i in categories]

    # 创建图形
    fig, ax = plt.subplots(figsize=(30, 10))

    # 绘制箱线图
    # 绘制箱线图（保留之前的优化设置）
    box1 = ax.boxplot(
        seeds,
        positions=positions_seeds,
        widths=0.3,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=2),
        medianprops=dict(linestyle='-', linewidth=2, color='black'),
    )

    box2 = ax.boxplot(
        others,
        positions=positions_others,
        widths=0.3,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=2),
        medianprops=dict(linestyle='-', linewidth=2, color='black'),
    )

    # 设置颜色和透明度
    for b in box1['boxes']:
        b.set_facecolor('skyblue')
        b.set_alpha(0.8)
    for b in box2['boxes']:
        b.set_facecolor('salmon')
        b.set_alpha(0.8)

    # 计算误差条数据（均值和标准差）
    # 对每个类别中的两个数据集分别计算
    means_seeds = [np.mean(d) for d in seeds]
    stds_seeds = [np.std(d) for d in seeds]

    means_others = [np.mean(d) for d in others]
    stds_others = [np.std(d) for d in others]

    # 添加误差条（使用errorbar函数）
    # 种子集的误差条
    ax.errorbar(
        x=positions_seeds,
        y=means_seeds,
        yerr=stds_seeds,
        fmt='o',  # 数据点用圆点标记
        color='darkblue',
        elinewidth=2,
        capsize=5,
        markersize=6,
        label='Seed Mean ± Std'  # 图例标签
    )

    # 其他样本的误差条
    ax.errorbar(
        x=positions_others,
        y=means_others,
        yerr=stds_others,
        fmt='s',  # 数据点用方块标记
        color='darkred',
        elinewidth=2,
        capsize=5,
        markersize=6,
        label='Other Mean ± Std'
    )

    # 设置Y轴范围和标签
    # ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks(categories)
    ax.set_xticklabels([f"Category {i+1}" for i in categories], fontsize=10)
    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Prediction Accuracy", fontsize=12)
    ax.set_title("Boxplot with Error Bars (Mean ± Std)", fontsize=14)

    # 更新图例（包含箱线图和误差条）
    legend_elements = [
        box1["boxes"][0],
        box2["boxes"][0],
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=8, label='Seed Mean'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', markersize=8, label='Other Mean')
    ]
    ax.legend(
        handles=legend_elements,
        labels=['Seed Box', 'Other Box', 'Seed Mean', 'Other Mean'],
        # loc='upper right',
        frameon=True
    )

    plt.tight_layout()

    plt.savefig("./imgs/3.png")
    return

if __name__ == "__main__":
    # 加载backdoor
    backdoor_data_path = os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            config.dataset_name, 
                                            config.model_name, 
                                            config.attack_name, 
                                            "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
    # 污染样本
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 加载pre_label_csv
    mutation_rate = 0.01
    pre_label_df = pd.read_csv(
        os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(mutation_rate),
            "preLabel.csv"
        )
    )
    # 变异模型数量
    mutation_model_num = 500
    # 验证一下id对应关系
    index_list = pre_label_df[pre_label_df["isPoisoned"] == True].index.to_list()
    assert set(poisoned_ids) == set(index_list), "id对应错误"
    total_num = pre_label_df.shape[0]
    # 所有样本id
    id_list = list(range(total_num))
    # 所有样本的真实分类标签
    gt_label_list = pre_label_df["GT_label"].tolist()
    main()