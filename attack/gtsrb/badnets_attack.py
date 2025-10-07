import os
import setproctitle
import torch
import torch.nn as nn
from attack.core.attacks import BadNets
from models.model_loader import get_model
from commonUtils import read_yaml
from attack.random_util import set_random_seed
from datasets.clean_dataset import get_clean_dataset
config = read_yaml("config.yaml")
global_random_seed = config["global_random_seed"]
set_random_seed(global_random_seed)
# CIFAR10+BedNets场景
dataset_name = "GTSRB"
attack_name = "BadNets"
model_name = "ResNet18"
target_class = config["target_class"]
poisoned_rate = config["poisoned_rate"]
gpu_id = 0
# clean dataset, victim model
clean_trainset,clean_testset = get_clean_dataset(dataset_name,attack_name)
model = get_model(dataset_name,model_name)
# Trigger
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0
# 攻击类
badnets = BadNets(
    train_dataset=clean_trainset,
    test_dataset=clean_testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=target_class,
    poisoned_rate=poisoned_rate,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index= -1,
    poisoned_transform_test_index= -1,
    poisoned_target_transform_index=0,
    seed=global_random_seed,
    deterministic=True
)
exp_root_dir = config["exp_root_dir"]
schedule = {
    'device': f'cuda:{gpu_id}',
    
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 150], # epoch区间 (150,180)

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}

def attack():
    # 攻击
    badnets.train(schedule)
    # 工作dir
    work_dir = badnets.work_dir
    # 获得backdoor model weights
    backdoor_model = badnets.best_model
    # poisoned trainset
    poisoned_trainset = badnets.poisoned_train_dataset
    # poisoned_ids
    poisoned_ids = poisoned_trainset.poisoned_set
    backdoor_data = {}
    backdoor_data["backdoor_model"] = backdoor_model
    backdoor_data["poisoned_ids"]=poisoned_ids
    backdoor_data["pattern"] = pattern
    backdoor_data['weight']=weight
    save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    torch.save(backdoor_data, save_path)
    print(f"BadNets攻击完成,数据被存入{save_path}")

if __name__ == "__main__":
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack()

    