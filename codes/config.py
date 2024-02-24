
dataset_name_list = ["CIFAR10","GTSRB", "MNIST"]
attack_name_list = ["BadNets", "Blended", "IAD", "LabelConsistent", "Refool", "WaNet"]
model_name_list = ["resnet18_nopretrain_32_32_3", "vgg19", "ResNet18", "VGG19", "DensNet", "BaselineMNISTNetwork", "CNN_Model_1"]
mutation_name_list = ["gf","neuron_activation_inverse","neuron_block","neuron_switch","weight_shuffle"]
mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
fine_mutation_rate_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
exp_root_dir = "/data/mml/backdoor_detect/experiments/"

dataset_name = "MNIST"
class_num = 10
model_name = "CNN_Model_1"
attack_name = "Refool"




