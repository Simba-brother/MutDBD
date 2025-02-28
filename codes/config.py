# 实验数据根目录
exp_root_dir = "/data/mml/backdoor_detect/experiments/"

# CIFAR-10 dataset dir
CIFAR10_dataset_dir = "/data/mml/backdoor_detect/dataset/cifar10"
# GTSRB dataset dir
GTSRB_dataset_dir = "/data/mml/backdoor_detect/dataset/GTSRB"
# ImageNet2012_subset dir
ImageNet2012_subset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"


# 随机种子
random_seed = 666
# 4个数据集名字
dataset_name_list = ["CIFAR10","GTSRB", "MNIST", "ImageNet2012_subset"]
cur_dataset_name_list = ["CIFAR10","GTSRB"]
# 6种攻击的名字
attack_name_list = ["BadNets", "Blended", "IAD", "LabelConsistent", "Refool", "WaNet"]
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
fine_mutation_rate_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
# 配置攻击类别
target_class_idx = 3
# 样本投毒比例
poisoned_rate = 0.05

###########################################################

# 当前实验设置的数据集名字
dataset_name = "ImageNet2012_subset" # CIFAR10, GTSRB, ImageNet2012_subset
# 数据集分类任务数量
class_num = 30 #  CIFAR-10:10,GTSRB:43,ImageNet2012_subset:30
# 当前实验设置的模型名字
model_name = "VGG19"
# 当前实验设置的攻击
attack_name = "WaNet"
# GPU设备
gpu_id = 1


# baseline ASD配置
asd_config = {
    "CIFAR10":{
        "epoch":120
    },
    "GTSRB":{
        "epoch":100
    },
    "ImageNetSub":{
        "epoch":120
    },
}






