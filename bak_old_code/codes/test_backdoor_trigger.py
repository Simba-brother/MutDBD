'''
对所有数据集，模型和攻击进行攻击测试。
'''
import os
import torch
from codes import config

from codes.transform_dataset import cifar10_BadNets,cifar10_IAD,cifar10_Refool,cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets,gtsrb_IAD,gtsrb_Refool,gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets,imagenet_IAD,imagenet_Refool,imagenet_WaNet

# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_poisoned_dataset as gtsrb_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_poisoned_dataset as gtsrb_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_poisoned_dataset as gtsrb_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_poisoned_dataset as gtsrb_WaNet_gen_poisoned_dataset

# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset as imagenetsub_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_poisoned_dataset as imagenetsub_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_poisoned_dataset as imagenetsub_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_poisoned_dataset as imagenetsub_WaNet_gen_poisoned_dataset

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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜的poisoned_testset
    poisoned_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")
    # 新鲜的poisoned_trainset
    poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test,"train")
    clean_trainset, clean_testset = cifar10_BadNets()

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

    evalModel = EvalModel(backdoor_model, clean_testset, device, batch_size=512, num_workers=4)
    print("clean_testset ACC:",evalModel.eval_acc())

    evalModel = EvalModel(backdoor_model, clean_trainset, device, batch_size=512, num_workers=4)
    print("clean_trainset ACC:",evalModel.eval_acc())

    evalModel = EvalModel(backdoor_model, poisoned_trainset, device, batch_size=512, num_workers=4)
    print("poisoned_trainset ACC:",evalModel.eval_acc())

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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜的测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜的测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    poisoned_trainset = cifar10_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"train")
    clean_trainset, _, clean_testset, _ = cifar10_IAD()
    device = torch.device(f"cuda:{config.gpu_id}")

    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

    evalModel = EvalModel(backdoor_model, poisoned_trainset, device, batch_size=512, num_workers=4)
    print("poisoned_trainset ACC:",evalModel.eval_acc())


    
    evalModel = EvalModel(backdoor_model, clean_trainset, device, batch_size=512, num_workers=4)
    print("clean_trainset ACC:",evalModel.eval_acc())

    evalModel = EvalModel(backdoor_model, clean_testset, device, batch_size=512, num_workers=4)
    print("clean_testset ACC:",evalModel.eval_acc())

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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
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
def GTSRB_ResNet18_BadNets():
    print("GTSRB_ResNet18_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "ResNet18", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_VGG19_BadNets():
    print("GTSRB_VGG19_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "VGG19", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def GTSRB_DenseNet_BadNets():
    print("GTSRB_DenseNet_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "DenseNet", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# IAD
def GTSRB_ResNet18_IAD():
    print("GTSRB_ResNet18_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "ResNet18", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_VGG19_IAD():
    print("GTSRB_VGG19_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "VGG19", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_IAD_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_DenseNet_IAD():
    print("GTSRB_DenseNet_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "DenseNet", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_IAD_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# Refool
def GTSRB_ResNet18_Refool():
    print("GTSRB_ResNet18_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "ResNet18", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_VGG19_Refool():
    print("GTSRB_VGG19_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "VGG19", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_DenseNet_Refool():
    print("GTSRB_DenseNet_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "DenseNet", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# WaNet
def GTSRB_ResNet18_WaNet():
    print("GTSRB_ResNet18_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "ResNet18", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_WaNet_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


def GTSRB_VGG19_WaNet():
    print("GTSRB_VGG19_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "VGG19", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_WaNet_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def GTSRB_DenseNet_WaNet():
    print("GTSRB_DenseNet_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "GTSRB", 
                                            "DenseNet", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = gtsrb_WaNet_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())


'''
ImageNet_sub
'''
# BadNets
def ImageNetsub_ResNet18_BadNets():
    print("ImageNetsub_ResNet18_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "ResNet18", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")
    # 新鲜训练集
    # poisoned_trainset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_train,"train")
    # poisoned_ids_train = backdoor_data["poisoned_ids"]

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_VGG19_BadNets():
    print("ImageNetsub_VGG19_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "VGG19", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")
    # 新鲜训练集
    # poisoned_trainset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_train,"train")
    # poisoned_ids_train = backdoor_data["poisoned_ids"]

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_DenseNet_BadNets():
    print("ImageNetsub_DenseNet_BadNets")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "DenseNet", 
                                            "BadNets", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_test,"test")
    # 新鲜训练集
    # poisoned_trainset = imagenetsub_badNets_gen_poisoned_dataset(poisoned_ids_train,"train")
    # poisoned_ids_train = backdoor_data["poisoned_ids"]

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# IAD
def ImageNetsub_ResNet18_IAD():
    print("ImageNetsub_ResNet18_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "ResNet18", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")
    
    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_VGG19_IAD():
    print("ImageNetsub_VGG19_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "VGG19", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_IAD_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")
    
    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_DenseNet_IAD():
    print("ImageNetsub_DenseNet_IAD")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "DenseNet", 
                                            "IAD", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_IAD_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")
    
    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# Refool
def ImageNetsub_ResNet18_Refool():
    print("ImageNet_ResNet18_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "ResNet18", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_VGG19_Refool():
    print("ImageNetsub_VGG19_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "VGG19", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_DenseNet_Refool():
    print("ImageNetsub_DenseNet_Refool")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "DenseNet", 
                                            "Refool", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_Refool_gen_poisoned_dataset(poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

# WaNet
def ImageNetsub_ResNet18_WaNet():
    print("ImageNetsub_ResNet18_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "ResNet18", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_WaNet_gen_poisoned_dataset("ResNet18",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=8)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=8)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_VGG19_WaNet():
    print("ImageNetsub_VGG19_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "VGG19", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_WaNet_gen_poisoned_dataset("VGG19",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=8)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=8)
    print("ASR:",evalModel.eval_acc())

def ImageNetsub_DenseNet_WaNet():
    print("ImageNetsub_DenseNet_WaNet")
    backdoor_data = torch.load(os.path.join(config.exp_root_dir, 
                                            "ATTACK", 
                                            "ImageNet2012_subset", 
                                            "DenseNet", 
                                            "WaNet", 
                                            "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids_train = backdoor_data["poisoned_ids"]
    # 预制的污染测试集
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]
    # 测试集的id
    poisoned_ids_test = list(range(len(poisoned_testset_fixed)))
    # 新鲜测试集
    poisoned_testset = imagenetsub_WaNet_gen_poisoned_dataset("DenseNet",poisoned_ids_test,"test")

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=8)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=8)
    print("ASR:",evalModel.eval_acc())

if __name__ == "__main__":
    CIFAR10_ResNet18_IAD()

