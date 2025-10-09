# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_needed_dataset as cifar10_badNets_needed_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_needed_dataset as cifar10_IAD_needed_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_needed_dataset as cifar10_Refool_needed_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_needed_dataset as cifar10_WaNet_needed_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_needed_dataset as gtsrb_badNets_needed_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_needed_dataset as gtsrb_IAD_needed_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_needed_dataset as gtsrb_Refool_needed_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_needed_dataset as gtsrb_WaNet_needed_dataset
# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_needed_dataset as imagenet_badNets_needed_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_needed_dataset as imagenet_IAD_needed_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_needed_dataset as imagenet_Refool_need_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_needed_dataset as imagenet_WaNet_needed_dataset

# transform数据集(clean)
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets, gtsrb_IAD, gtsrb_Refool, gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets, imagenet_IAD, imagenet_Refool, imagenet_WaNet



def get_clean_dataset(dataset_name, attack_name):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets": 
            clean_trainset, clean_testset = cifar10_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = cifar10_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = cifar10_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = cifar10_WaNet()
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets": 
            clean_trainset, clean_testset = gtsrb_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = gtsrb_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = gtsrb_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = gtsrb_WaNet()
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            clean_trainset, clean_testset = imagenet_BadNets()
        elif attack_name == "IAD":
            clean_trainset, _, clean_testset, _ = imagenet_IAD()
        elif attack_name == "Refool":
            clean_trainset, clean_testset = imagenet_Refool()
        elif attack_name == "WaNet":
            clean_trainset, clean_testset = imagenet_WaNet()
    return clean_trainset, clean_testset

def get_all_dataset(dataset_name, model_name, attack_name, trainset_poisoned_ids):

    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets": # BadNets中毒操作比较快
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = cifar10_badNets_needed_dataset(trainset_poisoned_ids)
        elif attack_name == "IAD": # 中毒操作较慢，而且中毒后没有数据处理步骤了
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = cifar10_IAD_needed_dataset(model_name, trainset_poisoned_ids)
        elif attack_name == "Refool":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = cifar10_Refool_needed_dataset(trainset_poisoned_ids)
        elif attack_name == "WaNet":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = cifar10_WaNet_needed_dataset(model_name,trainset_poisoned_ids)
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = gtsrb_badNets_needed_dataset(trainset_poisoned_ids)
        elif attack_name == "IAD":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = gtsrb_IAD_needed_dataset(model_name,trainset_poisoned_ids)
        elif attack_name == "Refool":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = gtsrb_Refool_needed_dataset(trainset_poisoned_ids)
        elif attack_name == "WaNet":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = gtsrb_WaNet_needed_dataset(model_name, trainset_poisoned_ids)
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = imagenet_badNets_needed_dataset(trainset_poisoned_ids)
        elif attack_name == "IAD":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = imagenet_IAD_needed_dataset(model_name,trainset_poisoned_ids)
        elif attack_name == "Refool":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = imagenet_Refool_need_dataset(trainset_poisoned_ids)
        elif attack_name == "WaNet":
            # 硬盘中的数据集信息
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = imagenet_WaNet_needed_dataset(model_name, trainset_poisoned_ids)
    return poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset