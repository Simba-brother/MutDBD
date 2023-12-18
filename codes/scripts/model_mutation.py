import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import joblib
sys.path.append("./")
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes import draw
from codes.utils import create_dir
from codes import config
from scipy import stats

dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
mutation_name = config.mutation_name 
mutation_name_list =  config.mutation_name_list


if dataset_name == "CIFAR10":
    if model_name == "resnet18_nopretrain_32_32_3":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import *
    if model_name == "vgg19":
        if attack_name == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_vgg19 import *
        if attack_name == "Blended":
            from codes.datasets.cifar10.attacks.Blended_vgg19 import *
        if attack_name == "IAD":
            from codes.datasets.cifar10.attacks.IAD_vgg19 import *
        if attack_name == "LabelConsistent":
            from codes.datasets.cifar10.attacks.LabelConsistent_vgg19 import *
        if attack_name == "Refool":
            from codes.datasets.cifar10.attacks.Refool_vgg19 import *
        if attack_name == "WaNet":
            from codes.datasets.cifar10.attacks.WaNet_vgg19 import *

dict_state = get_dict_state()
backdoor_model = dict_state["backdoor_model"]    

mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
mutation_model_num = 50

base_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/"
device = torch.device("cuda:1")

def gf_mutate():
    scale = 5
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,f"ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}")
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._gf_mut(scale)    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")


def neuron_activation_inverse_mutate():
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_activation_inverse()    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_block_mutate():
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}") 
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_block()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_switch_mutate():
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")  
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_switch()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def weight_shuffling():
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")  
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._weight_shuffling()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")


def eval_mutated_model():
    save_path = f"/data/mml/backdoor_detect/experiments/{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    poisoned_trainset = dict_state["poisoned_trainset"]
    scale = 5
    data = defaultdict(list)
    for mutation_ratio in tqdm(mutation_ratio_list):
        if mutation_name == "gf":
            work_dir = os.path.join(base_dir, f"ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}")        
        else:
            work_dir = os.path.join(base_dir, f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")        
        for m_i in range(mutation_model_num):
            state_dict = torch.load(os.path.join(work_dir, f"model_mutated_{m_i+1}.pth"), map_location="cpu")
            backdoor_model.load_state_dict(state_dict)
            evalModel = EvalModel(backdoor_model, poisoned_trainset, device)
            report = evalModel._eval_classes_acc()
            data[mutation_ratio].append(report)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}")
    joblib.dump(data, save_path)


def draw_box_single():
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    origin_report = e._eval_classes_acc()
    file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    file_path = os.path.join("/data/mml/backdoor_detect/experiments",file_name)
    data = joblib.load(file_path)
    ans = {}
    for mutation_ratio in mutation_ratio_list:
        ans[mutation_ratio] = {}
        for class_idx in range(10):
            ans[mutation_ratio][class_idx] = []
    for mutation_ratio in mutation_ratio_list:
        report_list = data[mutation_ratio]
        for report in report_list:
            for class_idx in range(10):
                precision = report[str(class_idx)]["precision"]
                origin_precison = origin_report[str(class_idx)]["precision"]
                dif = round(origin_precison-precision,3)
                ans[mutation_ratio][class_idx].append(dif)

    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box")
    create_dir(save_dir)
    for mutation_ratio in mutation_ratio_list:
        all_y = []
        labels = []
        for class_i in range(10):
            y_list = ans[mutation_ratio][class_i]
            all_y.append(y_list)
            labels.append(f"Class_{class_i}")
        title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}_{mutation_ratio}"
        save_file_name = title+".png"
        save_path = os.path.join(save_dir, save_file_name)
        draw.draw_box(all_y, labels,title,save_path)

def draw_box_all():
    # 数据集/模型/攻击名称
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    origin_report = e._eval_classes_acc()
    # 存各个变异方法
    data_list = []
    # 遍历变异方法
    for mutation_name in  config.mutation_name_list:
        file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
        file_path = os.path.join("/data/mml/backdoor_detect/experiments",file_name)
        data = joblib.load(file_path)
        data_list.append(data)
    ans = {}
    for mutation_ratio in mutation_ratio_list:
        ans[mutation_ratio] = {}
        for class_i in range(10):
            ans[mutation_ratio][class_i] = []
    for data in data_list:
        for mutation_ratio in mutation_ratio_list:
            report_list = data[mutation_ratio]
            for report in report_list:
                for class_i in range(10):
                    dif = origin_report[str(class_i)]["precision"] - report[str(class_i)]["precision"]
                    ans[mutation_ratio][class_i].append(dif)

    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box")
    for mutation_ratio in mutation_ratio_list:
        all_y = []
        labels = []
        for class_i in range(10):
            y_list = ans[mutation_ratio][class_i]
            all_y.append(y_list)
            labels.append(f"Class_{class_i}")
        title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_ratio}"
        save_file_name = title+".png"
        save_path = os.path.join(save_dir, save_file_name)
        draw.draw_box(all_y, labels,title,save_path)

def mutate():
    if mutation_name == "gf":
        gf_mutate()
    if mutation_name == "neuron_activation_inverse":
        neuron_activation_inverse_mutate()
    if mutation_name == "neuron_block":
        neuron_block_mutate()
    if mutation_name == "neuron_switch":
        neuron_switch_mutate()
    if mutation_name == "weight_shuffle":
        weight_shuffling()




    # p_dict = {}
    # for mutation_ratio in mutation_ratio_list:
    #     min_mean_class_i = -1
    #     min_median_class_i = -1
    #     min_mean = float('inf')
    #     min_median = float('inf')
    #     for class_i in range(10):
    #         if np.mean(ans[mutation_ratio][class_i]) < min_mean:
    #             min_mean_class_i = class_i
    #             min_mean = np.mean(ans[mutation_ratio][class_i])
    #         if np.median(ans[mutation_ratio][class_i]) < min_median:
    #             min_median_class_i = class_i
    #             min_median = np.median(ans[mutation_ratio][class_i])
        
    #     if min_mean_class_i == min_median_class_i:
    #         base_data = ans[mutation_ratio][min_mean_class_i]
    #         p_list = []
    #         for class_i in range(10):
    #             if class_i != min_mean_class_i:
    #                 target_data = ans[mutation_ratio][class_i]
    #                 res = stats.wilcoxon(base_data, target_data)
    #                 p_list.append(res.pvalue)
    #         p_dict[mutation_ratio] = p_list
    # return
if __name__ == "__main__":
    mutate()
    # eval_mutated_model()
    # draw_box_single()
    # draw_box_all()
    pass