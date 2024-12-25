'''
绘制检测中毒样本性能图
'''
import os
import joblib
from codes import config
import matplotlib.pyplot as plt
# 加载数据


def draw_line(x_ticks, title:str, xlabel:str, ylabel:str, save_path:str, draw_data_dict:dict):
    # 设置图片大小，清晰度
    # plt.figure(figsize=(20, 8), dpi=800)
    x_list = [x for x in list(range(len(x_ticks)))]
    for key,value in draw_data_dict.items():
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
    plt.legend()
    plt.show()
    plt.savefig(save_path,transparent=False,dpi=600)

draw_data = {}
for ratio in config.fine_mutation_rate_list:
    data_path = os.path.join(
        config.exp_root_dir,
        "DetectPoisonedSamples",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(ratio),
        "P_R_F1_dict.data"
    )
    P_R_F1_dict = joblib.load(data_path)
    cut_off_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    # draw_data[ratio]["precision_list"] = P_R_F1_dict["precision_list"]
    # draw_data[ratio]["recall_list"] = P_R_F1_dict["recall_list"]
    draw_data[ratio] = P_R_F1_dict["recall_list"]

save_dir = os.path.join(
    config.exp_root_dir,
    "Figures",
    config.dataset_name,
    config.model_name,
    config.attack_name,
)
os.makedirs(save_dir,exist_ok=True)
save_file_name = "Recall.png"
save_path = os.path.join(save_dir,save_file_name)
draw_line(
    x_ticks = cut_off_list, 
    title = "DetectPoisonedSamples", 
    xlabel="Cut Off",
    ylabel="Recall", 
    save_path=save_path, 
    draw_data_dict=draw_data)
