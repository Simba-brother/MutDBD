'''
选择top50变异模型
'''
import os
import queue
import torch
from sklearn.metrics import classification_report
import pandas as pd
from codes.utils import priorityQueue_2_list

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


def global_id_To_path():
    _map = {}
    mutation_operator_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
    mutation_model_num = 100
    global_model_id = 0
    for operator in mutation_operator_list:
        for i in range(mutation_model_num):
            _map[global_model_id] = os.path.join(exp_root_dir,"MutationModels",dataset_name,model_name,attack_name,str(mutated_rate),operator,f"model_{i}.pth")
            global_model_id += 1
    return _map



def main():
    df = pd.read_csv(os.path.join(
                exp_root_dir,
                "EvalMutationToCSV",
                dataset_name,
                model_name,
                attack_name,
                str(mutated_rate),
                "preLabel.csv")
        )
    selected_m_i_list = get_top_k_global_ids(df)
    _map = global_id_To_path()
    removed_m_i_set = set(list(range(500))) - set(selected_m_i_list)
    for removed_m_i in removed_m_i_set:
        model_path = _map[removed_m_i]
        os.remove(model_path)


if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "CIFAR10"
    model_name = "VGG19"
    attack_name = "LabelConsistent"
    # 获得backdoor_data
    backdoor_data_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name,attack_name, "backdoor_data.pth")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    mutated_rate_list = [0.03,0.05,0.07,0.09,0.1]
    for mutated_rate in mutated_rate_list:
        main()
    print("END")