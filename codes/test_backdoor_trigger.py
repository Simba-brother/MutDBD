import os
import torch
from codes import config
from codes.poisoned_dataset.cifar10.badNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.common.eval_model import EvalModel

# 加载后门攻击配套数据
def CIFAR10_ResNet18_BadNets():
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
    poisoned_testset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids_test)

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

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
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset("ResNet18",poisoned_ids_test)
    

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset_fixed, device, batch_size=512, num_workers=4)
    print("ASR_fixed:",evalModel.eval_acc())

    device = torch.device(f"cuda:{config.gpu_id}")
    evalModel = EvalModel(backdoor_model, poisoned_testset, device, batch_size=512, num_workers=4)
    print("ASR:",evalModel.eval_acc())

if __name__ == "__main__":
    CIFAR10_ResNet18_IAD()





