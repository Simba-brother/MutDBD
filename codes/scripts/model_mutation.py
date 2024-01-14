import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import joblib
import setproctitle
from torch.utils.data import DataLoader,Dataset
sys.path.append("./")
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes import draw
from codes.utils import create_dir
from codes import config



dataset_name = config.dataset_name
model_name = config.model_name
attack_name = config.attack_name
# 变异算子列表
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

# 攻击类别数据集
class TargetClassDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx  = target_class_idx
        self.target_class_dataset = self.get_target_class_dataset()

    def get_target_class_dataset(self):
        target_class_dataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id]
            if label == self.target_class_idx:
                target_class_dataset.append((sample, label))
        return target_class_dataset
    
    def __len__(self):
        return len(self.target_class_dataset)
    
    def __getitem__(self, index):
        x,y=self.target_class_dataset[index]
        return x,y

# 获得 数据集/模型/攻击 下的结果字典
dict_state = get_dict_state()
# 后门模型（带有后门权重）
backdoor_model = dict_state["backdoor_model"]    
# 从小到大的变异率
mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
# 变异算子/变异率,变异模型数量
mutation_model_num = 50

# 实验结果保存文件夹路径
exp_root_dir_path = "/data/mml/backdoor_detect/experiments"
# dataset/model/mutation operator
base_dir = os.path.join(exp_root_dir_path, dataset_name, model_name, "mutates")
# 设备
device = torch.device("cuda:1")

def gf_mutate(mutation_name="gf"):
    '''
    高斯模糊变异
    '''
    scale = 5
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir, mutation_name, f"ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}")
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._gf_mut(scale)    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_activation_inverse_mutate(mutation_name="neuron_activation_inverse"):
    '''
    神经元激活值翻转变异
    '''
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_activation_inverse()    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_block_mutate(mutation_name="neuron_block"):
    '''
    神经元阻塞变异
    '''
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}") 
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_block()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_switch_mutate(mutation_name="neuron_switch"):
    '''
    神经元切换变异
    '''
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")  
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_switch()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def weight_shuffling(mutation_name = "weight_shuffle"):
    '''
    权重打乱变异
    '''
    for mutation_ratio in mutation_ratio_list:
        work_dir = os.path.join(base_dir,mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")  
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._weight_shuffling()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = os.path.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def eval_mutated_model(mutation_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在整个后门训练集上各个分类上的report
    '''
    save_dir = os.path.join(exp_root_dir_path, dataset_name, model_name, attack_name, mutation_name)
    create_dir(save_dir)
    save_file_name = f"eval_poisoned_trainset_report.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 整个后门训练集
    poisoned_trainset = dict_state["poisoned_trainset"]
    scale = 5
    # {mutation_ratio:[acc_dict]}
    # acc_data = defaultdict(list)
    # {mutation_ratio:[report]}
    report_data = defaultdict(list)
    for mutation_ratio in tqdm(mutation_ratio_list):
        if mutation_name == "gf":
            work_dir = os.path.join(base_dir, mutation_name,f"ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}")        
        else:
            work_dir = os.path.join(base_dir, mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}")        
        for m_i in range(mutation_model_num):
            state_dict = torch.load(os.path.join(work_dir, f"model_mutated_{m_i+1}.pth"), map_location="cpu")
            backdoor_model.load_state_dict(state_dict)
            evalModel = EvalModel(backdoor_model, poisoned_trainset, device)
            report = evalModel._eval_classes_acc()
            report_data[mutation_ratio].append(report)
    joblib.dump(report_data, save_path)
    return report_data


def eval_mutated_model_in_target_class(mutation_name):
    '''
    dataset/model_name/attack_name/mutation operator
    评估在后门训练集上target class上(clean,posioned,整体)的accuracy
    '''
    save_dir = os.path.join(exp_root_dir_path, dataset_name, model_name, attack_name, mutation_name)
    save_file_name = f"eval_poisoned_trainset_target_class.data"
    save_path =  os.path.join(save_dir, save_file_name)
    # 目标类索引
    target_class_idx = 1
    # clean
    clean_set = dict_state["pureCleanTrainDataset"]
    # poisoned
    poisoned_set = dict_state["purePoisonedTrainDataset"]
    # whole
    whole_set = dict_state["poisoned_trainset"]
    # 把目标类数据集分为clean和poisoned
    clean_target = TargetClassDataset(clean_set, target_class_idx)
    poisoned_target = TargetClassDataset(poisoned_set, target_class_idx)
    whole_target = TargetClassDataset(whole_set, target_class_idx)
    # {mutation_ratio:[{"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole}]}
    res_dict = dict()
    scale = 5
    for mutation_ratio in tqdm(mutation_ratio_list):
        if mutation_name == "gf":
            mutation_models_dir = os.path.join(base_dir, mutation_name,f"ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}") 
        else:
            mutation_models_dir = os.path.join(base_dir, mutation_name,f"ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}") 
        # 获得该变异率下的变异模型的state_dict
        temp_list = []
        for m_i in range(50):
            mutation_model_state_dict_path = os.path.join(mutation_models_dir, f"model_mutated_{m_i+1}.pth")
            mutation_model_state_dict = torch.load(mutation_model_state_dict_path, map_location="cpu")
            backdoor_model.load_state_dict(mutation_model_state_dict)
            e = EvalModel(backdoor_model, clean_target, device)
            acc_clean = e._eval_acc()
            e = EvalModel(backdoor_model, poisoned_target, device)
            acc_poisoned = e._eval_acc()
            e = EvalModel(backdoor_model, whole_target, device)
            acc_whole = e._eval_acc()
            temp_list.append({"target_class_clean_acc":acc_clean, "target_class_poisoned_acc":acc_poisoned, "target_class_acc":acc_whole})
        res_dict[mutation_ratio] = temp_list
    joblib.dump(res_dict, save_path)
    return res_dict

def test(mutation_name):
    exp_root_dir_path = "/data/mml/backdoor_detect/experiments"
    save_path = f"{exp_root_dir_path}/{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    data_1 =  joblib.load(save_path)
    save_file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}_targetClass.data"
    save_path = os.path.join(exp_root_dir_path, save_file_name)
    data_2 =  joblib.load(save_path)
    print("fa")

def draw_box_by_mutaion_name(mutation_name):
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    origin_report = e._eval_classes_acc()
    # acc_dic_path = os.path.join(exp_root_dir_path, dataset_name, model_name, attack_name, mutation_name, "eval_poisoned_trainset_acc.data")
    # acc_dic_data = joblib.load(acc_dic_path)
    report_path = os.path.join(exp_root_dir_path, dataset_name, model_name, attack_name, mutation_name, "eval_poisoned_trainset_report.data")
    report_data = joblib.load(report_path)
    # acc_dif_res = {}
    # {mutation_ratio:{class_idx:[precision_dif]}}
    precision_res = {}
    for mutation_ratio in mutation_ratio_list:
        # acc_dif_res[mutation_ratio] = {}
        precision_res[mutation_ratio] = {}
        for class_idx in range(10):
            # acc_dif_res[mutation_ratio][class_idx] = []
            precision_res[mutation_ratio][class_idx] = []
    # for mutation_ratio in mutation_ratio_list:
    #     acc_dict_list = acc_dic_data[mutation_ratio]
    #     for acc_dict in acc_dict_list:
    #         for class_idx in range(10):
    #             acc = acc_dict[class_idx]
    #             origin_acc = origin_acc_dict[class_idx]
    #             acc_dif = round(origin_acc-acc,3)
    #             acc_dif_res[mutation_ratio][class_idx].append(acc_dif)
    for mutation_ratio in mutation_ratio_list:
        report_list = report_data[mutation_ratio]
        for report in report_list:
            for class_idx in range(10):
                precision = report[str(class_idx)]["precision"]
                origin_precision = origin_report[str(class_idx)]["precision"]
                precision_dif = round(origin_precision-precision,3)
                precision_res[mutation_ratio][class_idx].append(precision_dif)

    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box", dataset_name, model_name, attack_name, mutation_name, "precision_dif")
    create_dir(save_dir)
    # for mutation_ratio in mutation_ratio_list:
    #     all_y = []
    #     labels = []
    #     for class_i in range(10):
    #         y_list = acc_dif_res[mutation_ratio][class_i]
    #         all_y.append(y_list)
    #         labels.append(f"Class_{class_i}")
    #     title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}_{mutation_ratio}"
    #     save_file_name = title+"_acc_dif"+".png"
    #     save_path = os.path.join(save_dir, save_file_name)
    #     draw.draw_box(all_y, labels,title,save_path)

    for mutation_ratio in mutation_ratio_list:
        all_y = []
        labels = []
        for class_i in range(10):
            y_list = precision_res[mutation_ratio][class_i]
            all_y.append(y_list)
            labels.append(f"Class_{class_i}")
        
        title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}_{mutation_ratio}"
        save_file_name = title+"_precision_dif"+".png"
        save_path = os.path.join(save_dir, save_file_name)
        xlabel = "Category"
        ylabel = "Precision difference"
        draw.draw_box(all_y, labels, title, xlabel, ylabel, save_path)

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
        data_file_path = os.path.join(exp_root_dir_path, dataset_name, model_name, attack_name, mutation_name, "eval_poisoned_trainset_report.data")
        data = joblib.load(data_file_path)
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
                    precision_dif = origin_report[str(class_i)]["recall"] - report[str(class_i)]["recall"]
                    ans[mutation_ratio][class_i].append(precision_dif)

    save_dir = os.path.join("/data/mml/backdoor_detect/experiments", "images/box", dataset_name, model_name, attack_name, "all_mutation", "Accuracy_dif")
    create_dir(save_dir)
    for mutation_ratio in mutation_ratio_list:
        all_y = []
        labels = []
        for class_i in range(10):
            y_list = ans[mutation_ratio][class_i]
            all_y.append(y_list)
            labels.append(f"Class_{class_i}")
        title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_ratio}"
        save_file_name = title+"_accuracy_dif"+".png"
        save_path = os.path.join(save_dir, save_file_name)
        xlable = "Category"
        ylabel = "Accuracy difference"
        draw.draw_box(all_y, labels,title, xlable, ylabel,save_path)

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
    # 生成变异模型
    # mutate()

    # 在poisoned trainset上对变异模型进行评估
    # for mutation_name in tqdm(mutation_name_list[2:]):
    #     data = eval_mutated_model(mutation_name)
    #     print(f"dataset_name:{dataset_name}, model_name:{model_name}, attack_name:{attack_name}, mutation_name:{mutation_name}")

    # 在target class set上对变异模型进行评估
    # setproctitle.setproctitle(attack_name)
    # for mutation_name in tqdm(mutation_name_list):
    #     eval_mutated_model_in_target_class(mutation_name)
    #     print(f"dataset_name:{dataset_name}, model_name:{model_name}, attack_name:{attack_name}, mutation_name:{mutation_name}")

    # test_2()

    for mutation_name in tqdm(mutation_name_list):
        draw_box_by_mutaion_name(mutation_name)

    # draw_box_all()
    pass