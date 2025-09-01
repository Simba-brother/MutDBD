import os
from commonUtils import read_yaml
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
    backdoor_data = os.path.join(exp_root_dir,"ATTACK",dataset_name,model_name,attack_name,
            "backdoor_data.pth"
        )
    return backdoor_data