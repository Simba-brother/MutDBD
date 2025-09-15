'''
This is the test code of poisoned training under LabelConsistent.
'''
import os
import copy
import torch
import torch.nn as nn
import setproctitle
from attack.models import get_model
from datasets.clean_dataset import get_clean_dataset
from commonUtils import read_yaml,set_random_seed
from attack.core.attacks import LabelConsistent
from mid_data_loader import get_labelConsistent_benign_model
from torchvision import transforms
from torch.utils.data import DataLoader 
from collections import Counter

config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]
dataset_name = "GTSRB"
attack_name = "LabelConsistent"
img_size  = 32
model_name = "ResNet18"
is_benign = False
gpu_id = 0
target_class = config["target_class"]
global_random_seed = config["global_random_seed"]

set_random_seed(global_random_seed)
experiment_name = "benign_train" if is_benign else "attack_train"

def get_trigger():
    # 图片四角白点
    pattern = torch.zeros((img_size, img_size), dtype=torch.uint8)
    # pattern[:3,:3] = 255
    # pattern[:3,-3:] = 255
    # pattern[-3:,:3] = 255
    # pattern[-3:,-3:] = 255
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((img_size, img_size), dtype=torch.float32)
    weight[:3,:3] = 1.0
    weight[:3,-3:] = 1.0
    weight[-3:,:3] = 1.0
    weight[-3:,-3:] = 1.0
    return pattern,weight


schedule = {
    'device': f'cuda:{gpu_id}',

    'benign_training': is_benign,
    'batch_size': 256,
    'num_workers': 8,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [20],

    'epochs': 50,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': experiment_name
}


def get_attacker(trainset,testset,victim_model,attack_class,poisoned_rate,
                 adv_model,adv_dataset_dir):

    # pattern,weight = get_trigger() # CIFAR10用
    eps = 16 # Maximum perturbation for PGD adversarial attack. Default: 8. # 
    alpha = 1.5 # Step size for PGD adversarial attack. Default: 1.5.
    steps = 100 # Number of steps for PGD adversarial attack. Default: 100.
    max_pixel = 255
    # attack = torchattacks.PGD(model, eps = 0.3, alpha = 1/255, steps=40, random_start=False)
    print(f"eps:{eps},alpha:{alpha},steps:{steps}")
    attacker = LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=victim_model,
        adv_model=adv_model,
        adv_dataset_dir=adv_dataset_dir, # os.path.join(exp_root_dir,"ATTACK", dataset_name, model_name, attack_name, "adv_dataset", f"eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}"),
        loss=nn.CrossEntropyLoss(),
        y_target=attack_class,
        poisoned_rate=poisoned_rate,
        adv_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size,img_size)), transforms.ToTensor()]),
        # pattern=pattern,
        # weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=2,
        poisoned_transform_test_index=2,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_random_seed,
        deterministic=True
    )
    return attacker


def bengin_main(model,trainset,testset):
    poisoned_rate = 0
    adv_model = None
    adv_dataset_dir = None
    attacker = get_attacker(trainset,testset,model,target_class,poisoned_rate,
                            adv_model,adv_dataset_dir)
    attacker.train()
    print("bengin model save in:", os.path.join(attacker.work_dir, "best_model.pth"))
    return attacker.best_model

def check_labels(dataset):
    
    train_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
    labels = []
    for batch in train_loader:
        y = batch[1]
        labels.extend(y.tolist())
    print(Counter(labels))




def attack_main(model,trainset,testset):
    poisoned_rate = 0.1
    # 被对抗模型
    adv_model = copy.deepcopy(model)
    benign_state_dict = get_labelConsistent_benign_model(dataset_name,model_name)
    adv_model.load_state_dict(benign_state_dict)
    # 得到对抗数据集
    adv_dataset_dir = os.path.join(exp_root_dir,"ATTACK", dataset_name, model_name, attack_name, "adv_dataset")
    attacker = get_attacker(trainset,testset,model,target_class,poisoned_rate,
                            adv_model,adv_dataset_dir)
    # print("trainset")
    # check_labels(trainset)
    # print("testset")
    # check_labels(testset)
    # print("p_trainset")
    # check_labels(attacker.poisoned_train_dataset)
    # print("p_testset")
    # check_labels(attacker.poisoned_test_dataset)

    attacker.train()

    print("LC攻击结束,开始保存攻击数据")
    backdoor_model = attacker.best_model
    bd_res = {}
    # poisoned_testset = attacker.poisoned_test_dataset
    poisoned_ids = attacker.poisoned_set
    bd_res["backdoor_model"] = backdoor_model
    bd_res["poisoned_ids"] = poisoned_ids
    pattern,weight = get_trigger()
    bd_res["pattern"] = pattern
    bd_res["pattern"] = weight
    save_path = os.path.join(
        config["exp_root_dir"], "ATTACK",
        dataset_name, model_name, attack_name,
        "backdoor_data.pth")
    torch.save(bd_res, save_path)
    print(f"backdoor_data save in:{save_path}")
    return bd_res

def main():
    setproctitle.setproctitle(f"{dataset_name}|{model_name}|{attack_name}|attack")
    # 获得受害模型
    victim_model = get_model(dataset_name, model_name)
    # 获得数据集
    trainset,testset = get_clean_dataset(dataset_name,attack_name)
    if is_benign:
        benign_model = bengin_main(victim_model,trainset,testset)
    else:
        bd_res = attack_main(victim_model,trainset,testset)

if __name__ == "__main__":
    main()




