
data = {
    "CIFAR10_ResNet18_BadNets":{
        "PN_list":[49,25,13,7,4,2,1],
        "ASR_list":[0.012, 0.012, 0.012, 0.016, 0.015, 0.013, 0.013],
        "flag_list":[1,1,1,1,1,1,1]
    },
    "CIFAR10_VGG19_BadNets":{
        "PN_list":[61,31,16,8,4,2,1],
        "ASR_list":[0.937,0.015,0.015,0.017,0.016,0.015,0.018],
        "flag_list":[0,1,1,1,1,1,1]
    },
    "CIFAR10_DenseNet_BadNets":{
        "PN_list":[43,22,11,6,3,2,1],
        "ASR_list":[0.951,0.961,0.632,0.323,0.012,0.014,0.013],
        "flag_list":[0,0,0,0,1,1,1]
    },
    "GTSRB_ResNet18_BadNets":{
        "PN_list":[65,33,17,9,5,3,2,1],
        "ASR_list":[0.955,0.95,0.944,0.846,0.003,0.003,0.001,0.002],
        "flag_list":[0,0,0,0,1,1,1,1]
    },
    "GTSRB_VGG19_BadNets":{
        "PN_list":[80,40,20,10,5,3,2,1],
        "ASR_list":[0.958,0.951,0.935,0.825,0.003,0.002,0.003,0.003],
        "flag_list":[0,0,0,0,1,1,1,1]
    },
    "GTSRB_DenseNet_BadNets":{
        "PN_list":[78,39,20,10,5,3,2,1],
        "ASR_list":[0.965,0.952,0.946,0.887,0.003,0.003,0.005,0.002],
        "flag_list":[0,0,0,0,1,1,1,1]
    },
    "ImageNet_ResNet18_BadNets":{
        "PN_list":[47,24,12,6,3,2,1],
        "ASR_list":[0.803,0.778,0.001,0.002,0.001,0.003,0.001],
        "flag_list":[0,0,1,1,1,1,1]
    },
    "ImageNet_DenseNet_BadNets":{
        "PN_list":[69,35,18,9,5,3,2,1],
        "ASR_list":[0.932,0.906,0.657,0.001,0.001,0.001,0,0.001],
        "flag_list":[0,0,0,1,1,1,1,1]
    }
}


def draw_bar(PN_list, ASR_list, flag_list, save_path):
    '''
    绘制以PN_list为横轴，ASR_list为纵坐标的柱状图，flag_list为
    柱状图的颜色（1为绿0为红）save_path为图的保存路径(dpi为600)
    使用对数坐标以便更好地显示小数值
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Set scientific journal style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0

    # Create figure with appropriate size for journals (single column width)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

    # Prepare data
    x_pos = np.arange(len(PN_list))
    colors = ['#2ecc71' if flag == 1 else '#e74c3c' for flag in flag_list]

    # Replace 0 values with a small number for log scale
    ASR_list_plot = [max(asr, 1e-4) for asr in ASR_list]

    # Create bar chart
    bars = ax.bar(x_pos, ASR_list_plot, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    # Add value labels on top of each bar
    for i, (bar, asr) in enumerate(zip(bars, ASR_list)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{asr:.3f}',
                ha='center', va='bottom', fontsize=7, rotation=0)

    # Set labels and title
    ax.set_xlabel('Poisoned sample Number (PN)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PN_list, fontsize=9)

    # Use log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.2)

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5, which='both')
    ax.set_axisbelow(True)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
    plt.close()


def main(scence):
    PN_list = data[scence]["PN_list"]
    ASR_list = data[scence]["ASR_list"]
    flag_list = data[scence]["flag_list"]
    save_path  = f"imgs/dis_strip/{scence}.png"
    draw_bar(PN_list, ASR_list, flag_list, save_path)


if __name__ == "__main__":
    dataset_name_list = ["CIFAR10","GTSRB","ImageNet"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets"]
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            if dataset_name == "ImageNet" and model_name == "VGG19":
                continue
            for attack_name in attack_name_list:
                scence = f"{dataset_name}_{model_name}_{attack_name}"
                main(scence)