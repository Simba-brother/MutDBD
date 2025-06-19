'''
统计所有场景的W/T/L
'''
import os
import yaml
import pandas as pd
from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta


# 读取yaml
yaml_path = "codes/OurMethod_res_config.yaml"
with open(yaml_path, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "BadNets"
exp_id = "exp_1"
time_str = data[dataset_name][model_name][attack_name][exp_id]
# 获得模型路径
base_dir_path = "/data/mml/backdoor_detect/experiments/OurMethod/Retrain"
os.path.join(base_dir_path,dataset_name,model_name,attack_name,time_str, "last_BD_model.pth")
print("")





# 读取excel
excel_path = "/data/mml/backdoor_detect/experiments/实验结果/WTL.xlsx"
PN_dataframe = pd.read_excel(excel_path,sheet_name="PN")
for row_idx, row in PN_dataframe.iterrows():
     ASD_exp_list = [row["ASD_1"],row["ASD_2"],row["ASD_3"]]
     Our_exp_list = [row["Our_1"],row["Our_2"],row["Our_3"]]
     # 计算W/T/L
     # Wilcoxon:https://blog.csdn.net/TUTO_TUTO/article/details/138289291
     # Wilcoxon：主要来判断两组数据是否有显著性差异。
     statistic, p_value = wilcoxon(ASD_exp_list, Our_exp_list) # statistic:检验统计量
     # cliffs_delta：比较大小
     # 如果arg1 整体分布小则返回-1,反之为1
     d,res = cliffs_delta(ASD_exp_list, Our_exp_list)
print(PN_dataframe)


