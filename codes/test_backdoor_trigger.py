import os
import torch
from codes import config
from codes.poisoned_dataset.cifar10.badNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
from codes.common.eval_model import EvalModel


'''
CIFAR10
'''
# BadNets
def CIFAR10_ResNet18_BadNets():
    print("CIFAR10_ResNet18_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "ResNet18", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def CIFAR10_VGG19_BadNets():
    print("CIFAR10_VGG19_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "VGG19", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    poisoned_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def CIFAR10_DenseNet_BadNets():
    print("CIFAR10_DenseNet_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "DenseNet", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    poisoned_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


# IAD
def CIFAR10_ResNet18_IAD():
    print("CIFAR10_ResNet18_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "ResNet18", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def CIFAR10_VGG19_IAD():
    print("CIFAR10_VGG19_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "VGG19", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def CIFAR10_DenseNet_IAD():
    print("CIFAR10_DenseNet_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "DenseNet", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


# Refool
def CIFAR10_ResNet18_Refool():
    print("CIFAR10_ResNet18_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "ResNet18", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def CIFAR10_VGG19_Refool():
    print("CIFAR10_VGG19_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "VGG19", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def CIFAR10_DenseNet_Refool():
    print("CIFAR10_DenseNet_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "DenseNet", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# WaNet
def CIFAR10_ResNet18_WaNet():
    print("CIFAR10_ResNet18_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "ResNet18", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_WaNet_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def CIFAR10_VGG19_WaNet():
    print("CIFAR10_VGG19_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "VGG19", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_WaNet_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def CIFAR10_DenseNet_WaNet():
    print("CIFAR10_DenseNet_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "CIFAR10", 
                                            "DenseNet", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    # poisoned_trainset = gen_poisoned_dataset(poisoned_ids_train)
    poisoned_testset = cifar10_WaNet_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

'''
GTSRB
'''
# BadNets


if __name__ == "__main__":
    CIFAR10_ResNet18_WaNet()





