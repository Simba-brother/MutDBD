'''
分析变异模型在poisoned trainset的性能
'''
import os
import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from codes import config
from codes.common.eval_model import EvalModel
from codes.scripts.dataset_constructor import *


def get_mutated_models_eval_acc(mutated_models_eval_ans):
    data = {}
    for ratio in config.fine_mutation_rate_list:
        data[ratio] = []
        for op in config.mutation_name_list:
            for report in mutated_models_eval_ans[ratio][op]:
                data[ratio].append(report["accuracy"])
    ans = {}
    for ratio in config.fine_mutation_rate_list:
        avg = round(sum(data[ratio])/len(data[ratio]),4)
        ans[ratio] = avg
    return ans

def get_mutated_models_eval_acc_v2():
    data ={}
    for rate in config.fine_mutation_rate_list:
        # 加载变异模型评估结果(csv)
        df = pd.read_csv(os.path.join(
            config.exp_root_dir,
            "EvalMutationToCSV",
            config.dataset_name,
            config.model_name,
            config.attack_name,
            str(rate),
            "preLabel.csv"
        ))
        GT_label_list = list(df["GT_label"])
        avg_acc = 0
        for i in range(config.mutation_model_num*5):
            model_col_name = f"model_{i}"
            pred_label_list = list(df[model_col_name])
            acc = accuracy_score(GT_label_list,pred_label_list)
            avg_acc += acc
        avg_acc = round(avg_acc/(config.mutation_model_num*5),4)
        data[rate] = avg_acc
    return data

def draw_line(x_ticks_label,y_value_list, save_path):
    x_ticks = [0,2,4,6,8,10,12]

    plt.plot(x_ticks, y_value_list, label="Mutated models", marker='o')
    
    font_size=10
    plt.xticks(x_ticks,x_ticks_label,fontsize=font_size)

    plt.xlabel("mutation rate",fontsize=font_size)
    plt.tight_layout(pad=0)
    # 绘制网格(控制透明度)
    plt.grid(alpha=0.5, linestyle=':')
    # 添加图例
    plt.legend()
    plt.show()
    plt.savefig(save_path,transparent=False,dpi=600)



def main_v1():
    # 加载backdoor data
    # 加载后门模型数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    # 评估一下original backdoor model在poisoned_trainset上的acc
    device = torch.device(f"cuda:{config.gpu_id}")
    em = EvalModel(backdoor_model,poisoned_trainset,device)
    origin_acc = em.eval_acc()
    # 加载变异模型评估结果
    mutated_models_eval_ans = joblib.load(os.path.join(
        config.exp_root_dir,
        "EvalMutationResult_for_SuspiciousClasses",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "res.data"
    ))

    mutated_models_acc_dict = get_mutated_models_eval_acc(mutated_models_eval_ans)

    mutation_rate_list = config.fine_mutation_rate_list
    mutated_models_acc_list = []
    for rate in config.fine_mutation_rate_list:
        mutated_models_acc_list.append(mutated_models_acc_dict[rate])
    x_ticks_label = [0]
    x_ticks_label.extend(mutation_rate_list)
    
    y_value_list = [origin_acc]
    y_value_list.extend(mutated_models_acc_list)
    

    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "mutated_models_acc.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_line(x_ticks_label,y_value_list,save_path)
    print(f"save_path:{save_path}")

def main_v2():
    # 加载backdoor data
    # 加载后门模型数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    # 评估一下original backdoor model在poisoned_trainset上的acc
    device = torch.device(f"cuda:{config.gpu_id}")
    em = EvalModel(backdoor_model,poisoned_trainset,device)
    origin_acc = em.eval_acc()

    mutated_models_acc_dict = get_mutated_models_eval_acc_v2()

    mutation_rate_list = config.fine_mutation_rate_list
    mutated_models_acc_list = []
    for rate in config.fine_mutation_rate_list:
        mutated_models_acc_list.append(mutated_models_acc_dict[rate])
    x_ticks_label = [0]
    x_ticks_label.extend(mutation_rate_list)
    
    y_value_list = [origin_acc]
    y_value_list.extend(mutated_models_acc_list)
    

    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "mutated_models_acc.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_line(x_ticks_label,y_value_list,save_path)
    print(f"save_path:{save_path}")
    
if __name__ == "__main__":
    main_v1()