import os
import joblib
from codes import config
from collections import defaultdict
import matplotlib.pyplot as plt

def data_convertor(report_dict):
    res = defaultdict(list)
    for mutation_rate in report_dict.keys():
        for op in report_dict[mutation_rate].keys():
            for m_i in range(config.mutation_model_num):
                report = report_dict[mutation_rate][op][m_i]
                recall = report[str(config.target_class_idx)]["recall"]
                res[mutation_rate].append(recall)
    return res

def data_convertor_v2(data):
    res = {"clean":{},"poisoned":{}}
    for mutation_rate in data.keys():
        res["clean"][mutation_rate] = data[mutation_rate]["clean"]
        res["poisoned"][mutation_rate] = data[mutation_rate]["poisoned"]
    return res



def draw_box(clean_dict,poisoned_dict,save_path):
    
    clean_data = [clean_dict[rate] for rate in config.fine_mutation_rate_list]
    poisoned_data = [poisoned_dict[rate] for rate in config.fine_mutation_rate_list]
    # 设置箱线图宽度
    width = 0.2

    # 绘制箱线图
    plt.figure(figsize=(10, 8))

    xticks = [1,3,5,7,9,11]
    xticks_label = config.fine_mutation_rate_list
    # 绘制每个横坐标上的箱线图
    for i, (clean, poisoned) in enumerate(zip(clean_data, poisoned_data)):
        # 绘制 Clean 组
        plt.boxplot(clean, positions=[xticks[i] - width], widths=width, patch_artist=True, boxprops=dict(facecolor="skyblue", color="blue"), medianprops=dict(color="black"))
        
        # 绘制 Poisoned 组
        plt.boxplot(poisoned, positions=[xticks[i] + width], widths=width, patch_artist=True, boxprops=dict(facecolor="salmon", color="red"), medianprops=dict(color="black"))

    # 设置横轴标签
    # 设置横坐标刻度（控制刻度的范围和步长）
    plt.xticks(xticks,xticks_label)
    # 旋转横坐标标签，避免重叠
    plt.xticks(rotation=45)
    # 设置轴标签
    plt.xlabel('Mutation rate')
    plt.ylabel('Reccall')

    # 设置图例
    plt.legend(["Clean","Poisoned"])

    # 添加标题
    plt.title('Boxplot for Clean and Poisoned Data')

    # 显示图形
    plt.tight_layout(pad=0)
    # 绘制网格(控制透明度)
    plt.grid(alpha=0.5)
    plt.savefig(save_path,transparent=False,dpi=600)

def main_v1():
    # 加载数据
    clean_report_dict = joblib.load(os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "clean_report.data"
    ))

    poisoned_report_dict = joblib.load(os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "poisoned_report.data"
    ))
    
    clean_dict = data_convertor(clean_report_dict)
    poisoned_dict = data_convertor(poisoned_report_dict)

    # 绘制并保存
    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name,
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "TargetClass_clean_poisoned_recall_box.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_box(clean_dict,poisoned_dict,save_path)



def main_v2():
    # 加载数据
    data = joblib.load(os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "res.data"
    ))
    data_converted = data_convertor_v2(data)
    clean_dict = data_converted["clean"]
    poisoned_dict = data_converted["poisoned"]
    # 绘制并保存
    save_dir = os.path.join(
        config.exp_root_dir,
        "Figures",
        config.dataset_name,
        config.model_name,
        config.attack_name,
    )
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "TargetClass_clean_poisoned_recall_box.png"
    save_path = os.path.join(save_dir,save_file_name)
    draw_box(clean_dict,poisoned_dict,save_path)
    print(f"save_path:{save_path}")

if __name__ == "__main__":
    main_v2()
    


    