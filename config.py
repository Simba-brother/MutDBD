cur_config = {
    "dataset_name":"CIFAR-10", 
    "model_name":"ResNet18",
    "attack_name":"BadNets",
    "mutation_name":"GF" # // Gaussian Fuzzing
}



if cur_config["dataset_name"] == "CIFAR-10":
    if cur_config["model_name"] == "ResNet18":
        if cur_config["attack_name"] == "BadNets":
            from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
