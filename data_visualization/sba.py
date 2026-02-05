

import os
import pandas as pd
import numpy as np
from defense.our.mutation.mutation_select import get_top_k_global_ids
from utils.dataset_utils import get_class_num
from utils.small_utils import nested_defaultdict
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def draw_bar(data,class_num,save_path):
    classes = []
    fps = []
    for cls in range(class_num):
        classes.append(cls)
        fps.append(data[cls])
    

    # Colors: green for most, red for class 3
    colors = ["#2f8f2f"] * 10
    colors[3] = "#ff2b2b"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=200)

    bars = ax.bar(classes, fps, color=colors, edgecolor="black", linewidth=1.6, width=0.78)

    # Axes labels
    ax.set_xlabel("Class", fontsize=28)
    ax.set_ylabel("FPS", fontsize=28)

    # Limits and ticks (match the shown scale)
    # ax.set_ylim(0, 1426)
    # ax.set_yticks([0, 357, 713, 1070, 1426])
    ax.set_xticks(classes)

    # Grid (horizontal dotted lines)
    ax.grid(axis="y", linestyle=":", linewidth=1.3, color="0.7")

    # Minor ticks and tick style (ticks on all sides, inward)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", direction="in", top=True, right=True,
                length=9, width=2, labelsize=22, pad=10)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                length=5, width=1.5)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Value labels on top of bars
    for b, v in zip(bars, fps):
        ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:.1f}",
                ha="center", va="bottom", fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path,dpi=800)



def FP_metrics(df:pd.DataFrame,mutated_model_global_id_list:list, class_num):
    '''从大到小'''
    data_stuct_1 = nested_defaultdict(2,int)
    for m_i in mutated_model_global_id_list:
        pre_labels = df[f"model_{m_i}"]
        gt_labels = df[f"GT_label"]
        cm = confusion_matrix(gt_labels, pre_labels)
        for class_i in range(class_num):
            data_stuct_1[m_i][class_i] = np.sum(cm[:,class_i]) - cm[class_i][class_i]

    data_stuct_2 = nested_defaultdict(1,int)
    for class_i in range(class_num):
        for m_i in mutated_model_global_id_list:
            data_stuct_2[class_i] += data_stuct_1[m_i][class_i]
    
    for class_i in range(class_num):
        data_stuct_2[class_i] = round(data_stuct_2[class_i] / len(mutated_model_global_id_list),1)
    return data_stuct_2

def one_scene(dataset_name, model_name, attack_name, mutation_rate=0.01, metric="FP", save_dir=None):
    '''获得class rank list and rank top'''
    df_predicted_labels = pd.read_csv(os.path.join(exp_root_dir,"EvalMutationToCSV",dataset_name,model_name,attack_name,str(mutation_rate),"preLabel.csv"))
    mutated_model_id_list = get_top_k_global_ids(df_predicted_labels,top_k=50,trend="bigger")
    class_num = get_class_num(dataset_name)
    if metric == "FP":
        res = FP_metrics(df_predicted_labels, mutated_model_id_list, class_num)
    print(res)
    save_path = "imgs/SBA/cifar10_resnet18_sba.png"
    draw_bar(res,class_num,save_path)
    print(save_path)

if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    target_class = 3
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "SBA"
    one_scene(dataset_name, model_name, attack_name)