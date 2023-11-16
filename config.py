cur_config = {
    "dataset":"CIFAR-10", 
    "model":"ResNet18",
    "attack":"BadNets",
    "fangyu":"GF" # // Gaussian Fuzzing
}



dataset_list:["CIFAR-10", "MNIST", "GTSRB"]
dataset_list:["resnet18_pretrain", "resnet18_nopretrain_32_32_3"]
attack:["BadNets","WaNet"]
defense:{
    "mutation":['GF']
}