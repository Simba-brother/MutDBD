'''
绘制检测中毒样本性能图
'''
import os
import joblib
from codes import config
from codes.ourMethod.draw.common import draw_line
from collections import defaultdict
# 加载数据


draw_data = defaultdict(list)

data_path = os.path.join(
    config.exp_root_dir,
    "DetectPoisonedSamples_analyse",
    config.dataset_name,
    config.model_name,
    config.attack_name,
    "entropy.data"
)
data = joblib.load(data_path)
for ratio in config.fine_mutation_rate_list:
    draw_data["clean"].append(data[ratio]["clean_entropy"])
    draw_data["poisioned"].append(data[ratio]["poisoned_entropy"])

save_dir = os.path.join(
    config.exp_root_dir,
    "Figures",
    config.dataset_name,
    config.model_name,
    config.attack_name,
)
os.makedirs(save_dir,exist_ok=True)
save_file_name = "Clean_Poisoned_compareEntropy.png"
save_path = os.path.join(save_dir,save_file_name)
draw_line(
    x_ticks = config.fine_mutation_rate_list, 
    title = "DetectPoisonedSamples_analyse", 
    xlabel="mutation rate",
    ylabel="Entropy", 
    save_path=save_path, 
    draw_data_dict=draw_data)
