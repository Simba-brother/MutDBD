'''
用于中间数据加载
'''
import os
from utils.common_utils import read_yaml
import torch
import joblib


config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]

def get_CIFAR10_IAD_attack_dict_path(model_name):
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","ResNet18","IAD",
                                    "ATTACK_2024-12-18_13:17:49",
                                    "dict_state.pth")
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","VGG19","IAD",
                            "ATTACK_2024-12-18_13:20:48",
                            "dict_state.pth")
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","DenseNet","IAD",
                            "ATTACK_2024-12-18_13:24:29",
                            "dict_state.pth")
    return attack_dict_path

def get_GTSRB_IAD_attack_dict_path(model_name):
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB",model_name,"IAD",
                                    "ATTACK_2024-12-26_11:06:15",
                                    "dict_state.pth")
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB",model_name,"IAD",
                            "ATTACK_2024-12-26_11:06:59",
                            "dict_state.pth")
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB",model_name,"IAD",
                            "ATTACK_2024-12-26_21:31:24",
                            "dict_state.pth")
    return attack_dict_path

def get_CIFAR10_WaNet_attack_dict_path(model_name):
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","ResNet18","WaNet",
            "ATTACK_2024-12-18_13:37:18",
            "dict_state.pth"
        )
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","VGG19","WaNet",
            "ATTACK_2024-12-18_13:39:20",
            "dict_state.pth"
        )
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","CIFAR10","DenseNet","WaNet",
            "ATTACK_2024-12-18_13:41:03",
            "dict_state.pth"
        )
    return attack_dict_path

def get_GTSRB_WaNet_attack_dict_path(model_name):
    if model_name == "ResNet18":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB","ResNet18","WaNet",
            "ATTACK_2024-12-27_13:36:56",
            "dict_state.pth"
        )
    elif model_name == "VGG19":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB","VGG19","WaNet",
            "ATTACK_2024-12-27_13:37:37",
            "dict_state.pth"
        )
    elif model_name == "DenseNet":
        attack_dict_path = os.path.join(exp_root_dir,"ATTACK","GTSRB","DenseNet","WaNet",
            "ATTACK_2024-12-27_13:37:50",
            "dict_state.pth"
        )
    return attack_dict_path

def get_backdoor_data(dataset_name,model_name,attack_name):
    '''
    获得所有场景（dataset+model+attack）的后门数据
    '''
    backdoor_data_path = os.path.join(exp_root_dir,"ATTACK",dataset_name,model_name,attack_name,
            "backdoor_data.pth"
    )
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")

    return backdoor_data

def update_backdoor_data(backdoor_data, origin_backdoor_path):
    '''
    new_backdoor_data = {}
    new_backdoor_data[] = backdoor_data[]
    torch.save(new_backdoor_data, origin_backdoor_path)
    return new_backdoor_data
    '''
    pass


def get_class_rank(dataset_name,model_name,attack_name):
    '''
    获得所有场景（dataset+model+attack）的类别排序信息
    '''
    if attack_name != "LabelConsistent":
        data_path = os.path.join(exp_root_dir,"Exp_Results","ClassRank",dataset_name,model_name,attack_name,"res.joblib")
    else:
        data_path = os.path.join(exp_root_dir,"Exp_Results","ClassRank",dataset_name,model_name,attack_name,str(0.01),"FP.joblib")
    data = joblib.load(data_path)
    return data["class_rank"]

def get_our_method_state(dataset_name, model_name, attack_name, random_seed):
    '''获得我们方法fine_tuned_backdoor_model和defensed_model'''
    defensed_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours", dataset_name, model_name, attack_name, f"exp_{random_seed}", "last_defense_model.pth") 
    selected_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours", dataset_name, model_name, attack_name, f"exp_{random_seed}", "best_BD_model.pth") 
    return defensed_state_dict_path, selected_state_dict_path

def get_asd_method_state(dataset_name, model_name, attack_name, random_seed):

    '''获得ASD方法的fine_tuned_backdoor_model和defensed_model'''
    data = read_yaml("ASD_res_config.yaml")
    exp_id = f"exp_{random_seed}"
    time_str = data[dataset_name][model_name][attack_name][exp_id]
    # 获得防御模型路径
    defensed_state_dict_path = os.path.join(exp_root_dir,"Defense","ASD",dataset_name,model_name,attack_name,time_str,"ckpt","latest_model.pt") # key:"model_state_dict"
    # 获得选择模型路径
    selected_state_dict_path = os.path.join(exp_root_dir,"Defense","ASD",dataset_name,model_name,attack_name,time_str,"ckpt","secondtolast.pth")
    return defensed_state_dict_path, selected_state_dict_path

def get_labelConsistent_benign_model(dataset_name, model_name):
    
    LC_benign_record_dict = {
        "CIFAR10":{
            "ResNet18":"benign_train_2025-07-16_13:17:28",
            "VGG19":"benign_train_2025-07-16_17:35:57",
            "DenseNet":"benign_train_2025-07-16_22:34:07"
        },
        "GTSRB":{
            "ResNet18":"benign_model",
            "VGG19":"benign_train_2025-09-11_11:21:10",
            "DenseNet":"benign_train_2025-09-11_11:23:15" 
        }
    }
    key_dir = LC_benign_record_dict[dataset_name][model_name]
    benign_state_dict_path = os.path.join(exp_root_dir,"ATTACK",dataset_name, model_name, "LabelConsistent", key_dir, "best_model.pth")
    benign_state_dict = torch.load(benign_state_dict_path,map_location="cpu")
    return benign_state_dict



