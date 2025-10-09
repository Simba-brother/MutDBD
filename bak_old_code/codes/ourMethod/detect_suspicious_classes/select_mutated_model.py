'''
重要
从某个变异率下的变异模型中选择出与original backdoor model性能最接近的top 50个模型
'''
import os
import queue
import pandas as pd
from sklearn.metrics import classification_report
from codes.utils import priorityQueue_2_list
from codes import config



def get_top_k_global_ids(df:pd.DataFrame,top_k=50,trend="bigger"):
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    GT_labels = df["GT_label"]
    preLabels_o = df["original_backdoorModel_preLabel"]
    report_o = classification_report(GT_labels,preLabels_o,output_dict=True,zero_division=0)
    for m_i in range(500):
        col_name = f"model_{m_i}"
        preLabel_m = df[col_name]
        report_m = classification_report(GT_labels,preLabel_m,output_dict=True,zero_division=0)
        acc_dif = abs(report_o["accuracy"] - report_m["accuracy"])
        if trend == "smaller":
            item = (acc_dif, m_i)
        else:
            item = (-acc_dif, m_i)
        q.put(item)
    
    priority_list = priorityQueue_2_list(q)
    selected_m_i_list = [m_i for priority, m_i in  priority_list[0:top_k]]
    return selected_m_i_list

if __name__ == "__main__":
    rate = 0.05
    csv_path = os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        str(rate),
        "preLabel.csv"
        )
    df = pd.read_csv(csv_path)
    selected_m_i_list = get_top_k_global_ids(df,top_k=50)
