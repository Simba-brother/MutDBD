# 实验数据根目录
exp_root_dir = "/data/mml/backdoor_detect/experiments/"
# 随机种子
random_seed = 666
# 4个数据集名字
dataset_name_list = ["CIFAR10","GTSRB", "MNIST", "ImageNet"]
# 6种攻击的名字
attack_name_list = ["BadNets", "Blended", "IAD", "LabelConsistent", "Refool", "WaNet"]
# 模型名字
model_name_list = ["ResNet18", "VGG19", "DenseNet", "BaselineMNISTNetwork", "CNN_Model_1"]
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
dataset_name = "CIFAR10"
# 数据集分类任务数量
class_num = 10 # 
# 当前实验设置的模型名字
model_name = "DenseNet"
# 当前实验设置的攻击
attack_name = "WaNet"
# GPU设备
gpu_id = 0


# CIFAR-10 dataset dir
CIFAR10_dataset_dir = "/data/mml/backdoor_detect/dataset/cifar10"
GTSRB_dataset_dir = "/data/mml/backdoor_detect/dataset/GTSRB"

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






