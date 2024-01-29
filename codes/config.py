
dataset_name_list = ["CIFAR10","GTSRB", "MNIST"]
attack_name_list = ["BadNets", "Blended", "IAD", "LabelConsistent", "Refool", "WaNet"]
model_name_list = ["resnet18_nopretrain_32_32_3", "vgg19", "ResNet18", "VGG19", "DensNet", "BaselineMNISTNetwork", "CNN_Model_1"]
mutation_name_list = ["gf","neuron_activation_inverse","neuron_block","neuron_switch","weight_shuffle"]
mutation_rate_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
exp_root_dir = "/data/mml/backdoor_detect/experiments/"

dataset_name = "MNIST"
model_name = "CNN_Model_1"
attack_name = "WaNet"




