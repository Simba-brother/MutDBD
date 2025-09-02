import os
import torch
import torch.nn as nn

from attack.core.attacks import Refool
import setproctitle
from commonUtils import read_yaml
from attack.random_util import set_random_seed
from attack.models import get_model
from datasets.clean_dataset import get_clean_dataset
from attack.refool_util import get_reflection_images

config = read_yaml("config.yaml")
global_random_seed = config["global_random_seed"]
set_random_seed(global_random_seed)
dataset_name = "GTSRB"
model_name = "ResNet18"
attack_name = "Refool"
target_class = config["target_class"]
poisoned_rate = config["poisoned_rate"]
gpu_id = 1

clean_trainset,clean_testset = get_clean_dataset(dataset_name,attack_name)
model = get_model(dataset_name,model_name)
reflection_images = get_reflection_images()
# 攻击类实例
refool= Refool(
    train_dataset=clean_trainset,
    test_dataset=clean_testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=config.target_class_idx,
    poisoned_rate=config.poisoned_rate,
    poisoned_transform_train_index= 0,
    poisoned_transform_test_index= 0,
    poisoned_target_transform_index= 0,
    schedule=None,
    seed=global_random_seed,
    deterministic=True,
    reflection_candidates = reflection_images,
)
exp_root_dir = config["exp_root_dir"]
schedule = {
    'device': f'cuda:{gpu_id}',
    'GPU_num': 4,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],

    'epochs': 100,

    'log_iteration_interval': 100, # batch
    'test_epoch_interval': 10, # epoch
    'save_epoch_interval': 10,  # epoch

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}

def attack():
    print("开始attack train")
    refool.train(schedule)
    print("attack train结束")
    work_dir = refool.work_dir
    poisoned_trainset = refool.poisoned_train_dataset
    poisoned_ids = poisoned_trainset.poisoned_set
    # 获得backdoor model weights
    backdoor_weights = torch.load(os.path.join(work_dir, "best_model.pth"), map_location="cpu")
    # backdoor model存入字典数据中
    model.load_state_dict(backdoor_weights)
    backdoor_data = {}
    backdoor_data["backdoor_model"] = model
    backdoor_data["poisoned_ids"]=poisoned_ids
    save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    torch.save(backdoor_data, save_path)
    print(f"Refool攻击完成,数据被存入{save_path}")

if __name__ == "__main__":
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack()