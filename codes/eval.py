import os


import matplotlib.pyplot as plt

from codes import config
from codes.common.eval_model import EvalModel
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.models import get_model


import torch
# 加载后门攻击配套数据
backdoor_data_path = os.path.join(config.exp_root_dir, 
                                        "ATTACK", 
                                        config.dataset_name, 
                                        config.model_name, 
                                        config.attack_name, 
                                        "backdoor_data.pth")
backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
# 后门模型
backdoor_model = backdoor_data["backdoor_model"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_testset = backdoor_data["poisoned_testset"] # 预制的poisoned_testset

# 干净测试集
clean_trainset, clean_testset = cifar10_BadNets()

# 获得原生模型
model = get_model(dataset_name=config.dataset_name, model_name=config.model_name)

# 加载模型权重
# 权重路径
dict_path = "/data/mml/backdoor_detect/experiments/OurMethod/CIFAR10/VGG19/BadNets/2025-03-23_20:54:05/ckpt/latest_model.pt"
dict_data = torch.load(dict_path,map_location="cpu")
# 权重load到模型中
model.load_state_dict(dict_data["model_state_dict"])
# 设备
device = torch.device(f"cuda:{config.gpu_id}")
em = EvalModel(model,clean_testset,device)
report =  em.eval_classification_report()
accuracy =  report["accuracy"]
print("acc:",accuracy)
label_list = list(range(config.class_num))
recall_list = []
for label in label_list:
   recall_list.append(report[str(label)]["recall"])

bars = plt.bar(
        x=label_list,
        height=recall_list,
        color='skyblue',
        edgecolor='black'
    )

# 在柱顶显示百分比（保留1位小数）
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,  # 横坐标居中
        height + 0.02,  # 纵坐标偏移（根据比例调整）
        f'{height:.1%}',  # 显示百分比格式
        ha='center', va='bottom',  # 水平居中，底部对齐
        fontsize=9
    )

plt.title('Accuracy of each category', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(ticks=label_list)  # 显式指定坐标轴刻度
plt.ylim(0, 1.3)  # 扩大Y轴范围避免文字溢出
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("imgs/4.png", bbox_inches='tight', pad_inches=0.0, dpi=800)
plt.close()
print("fjal")

