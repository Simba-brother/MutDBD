# 实验数据根目录
exp_root_dir = "/data/mml/backdoor_detect/experiments/"

# CIFAR-10 dataset dir
CIFAR10_dataset_dir = "/data/mml/backdoor_detect/dataset/cifar10"
# GTSRB dataset dir
GTSRB_dataset_dir = "/data/mml/backdoor_detect/dataset/GTSRB"
# ImageNet2012_subset dir
ImageNet2012_subset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"


# 随机种子
random_seed = 0
# 4个数据集名字
dataset_name_list = ["CIFAR10","GTSRB","ImageNet2012_subset"] # , "MNIST"
cur_dataset_name_list = ["CIFAR10","GTSRB"]
# 6种攻击的名字
attack_name_list = ["BadNets", "IAD", "Refool", "WaNet"] # "Blended","LabelConsistent"
cur_attack_name_list = ["BadNets","IAD","Refool","WaNet"]
# 模型名字
model_name_list = ["ResNet18", "VGG19", "DenseNet", "BaselineMNISTNetwork", "CNN_Model_1"]
cur_model_name_list = ["ResNet18","VGG19","DenseNet"]
# 5种模型级别变异算子
# mutation_name_list = ["gf","neuron_activation_inverse","neuron_block","neuron_switch","weight_shuffle"]
mutation_name_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
# 变异比例list
mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8] # 0.001
# 每个变异率下的每个变异算子生成100个变异模型
mutation_model_num = 100 
# 更为精细化的变异比例
fine_mutation_rate_list = [0.01] # [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
# 配置攻击类别
target_class_idx = 3
# 样本投毒比例
poisoned_rate = 0.05

###########################################################

# 当前实验设置的数据集名字
dataset_name = "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
# 数据集分类任务数量
class_num = 10 #  CIFAR-10:10,GTSRB:43,ImageNet2012_subset:30
# 当前实验设置的模型名字
model_name = "ResNet18" # ResNet18, VGG19, DenseNet
# 当前实验设置的攻击
attack_name = "BadNets" # BadNets, IAD, Refool, WaNet
# GPU设备
gpu_id = 0
# baseline ASD配置
asd_config = {
    "CIFAR10":{
        "epoch":120
    },
    "GTSRB":{ 
        "epoch":100
    },
    "ImageNet2012_subset":{
        "epoch":120
    },
}

LC_attack_config = {
    "CIFAR10":{
        "ResNet18":{
            "benign_model_state_dir":"2025-06-23_12:06:52_2025-06-23_12:20:37"
        }
    }
}


'''
asd_result = {
    "CIFAR10":{
        "ResNet18":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/ResNet18/BadNets/2025-02-04_13:08:56/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/ResNet18/IAD/2025-02-17_17:27:09/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/ResNet18/Refool/2025-02-17_17:49:37/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/ResNet18/WaNet/2025-02-17_18:19:07/ckpt/latest_model.pt"}
        },
        "VGG19":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/VGG19/BadNets/2025-02-17_18:25:45/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/VGG19/IAD/2025-02-17_18:29:21/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/VGG19/Refool/2025-02-27_11:52:39/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/VGG19/WaNet/2025-02-17_18:34:34/ckpt/latest_model.pt"}
        },
        "DenseNet":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/DenseNet/BadNets/2025-02-17_18:39:21/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/DenseNet/IAD/2025-02-17_18:41:23/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/DenseNet/Refool/2025-02-17_18:47:05/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/CIFAR10/DenseNet/WaNet/2025-02-17_18:59:49/ckpt/latest_model.pt"}
        }
    },
    "GTSRB":{
        "ResNet18":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/ResNet18/BadNets/2025-02-17_18:54:05/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/ResNet18/IAD/2025-02-17_18:57:13/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/ResNet18/Refool/2025-02-17_19:40:32/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/ResNet18/WaNet/2025-02-17_20:01:40/ckpt/latest_model.pt"}
        },
        "VGG19":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/VGG19/BadNets/2025-02-17_20:02:49/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/VGG19/IAD/2025-02-17_20:05:46/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/VGG19/Refool/2025-02-17_20:28:34/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/VGG19/WaNet/2025-02-17_20:31:08/ckpt/latest_model.pt"}
        },
        "DenseNet":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/DenseNet/BadNets/2025-02-17_20:32:50/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/DenseNet/IAD/2025-02-17_20:35:23/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/DenseNet/Refool/2025-02-17_20:40:02/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/GTSRB/DenseNet/WaNet/2025-02-17_20:44:25/ckpt/latest_model.pt"}
        }
    },
    "ImageNet2012_subset":{
        "ResNet18":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/ResNet18/BadNets/2025-03-07_13:55:26/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/ResNet18/IAD/2025-03-07_20:28:50/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/ResNet18/Refool/2025-03-07_21:03:57/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/ResNet18/WaNet/2025-03-07_21:31:50/ckpt/latest_model.pt"}
        },
        "VGG19":{
            "BadNets":{"latest_model":""},
            "IAD":{"latest_model":""},
            "Refool":{"latest_model":""},
            "WaNet":{"latest_model":""}
        },
        "DenseNet":{
            "BadNets":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/DenseNet/BadNets/2025-03-12_11:30:35/ckpt/latest_model.pt"},
            "IAD":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/DenseNet/IAD/2025-03-12_11:40:01/ckpt/latest_model.pt"},
            "Refool":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/DenseNet/Refool/2025-03-14_11:10:26/ckpt/latest_model.pt"},
            "WaNet":{"latest_model":"/data/mml/backdoor_detect/experiments/ASD/ImageNet2012_subset/DenseNet/WaNet/2025-03-14_12:34:37/ckpt/latest_model.pt"}
        }
    }
}
'''



