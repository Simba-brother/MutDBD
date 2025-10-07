'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''
import os
import setproctitle
import torch
import torch.nn as nn
from attack.core.attacks import WaNet
from commonUtils import read_yaml
from datasets.clean_dataset import get_clean_dataset
from models.model_loader import get_model
from commonUtils import set_random_seed
config = read_yaml("config.yaml")
global_random_seed = config["global_random_seed"]
set_random_seed(global_random_seed)
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "Refool"
target_class = config["tareget_class"]
poisoned_rate = config["poisoned_rate"]
gpu_id = 1
clean_trainset,clean_testset = get_clean_dataset(dataset_name,attack_name)
model = get_model(dataset_name,model_name)

# Trigger
def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    height = 32
    k = 4
    """
    # shape:(1, 2, k, k), 均匀分布 从区间[0,1)的均匀分布中随机抽取 ndarray
    ins = torch.rand(1, 2, k, k) * 2 - 1 # 区间变为（-1，1）
    # 先去取tensor的绝对值=>均值=>所有数据再去除以均值
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

identity_grid,noise_grid=gen_grid(32,4)
# 攻击类实例
wanet = WaNet(
    train_dataset=clean_trainset,
    test_dataset=clean_testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=target_class,
    poisoned_rate=poisoned_rate,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    poisoned_transform_train_index=-3,
    poisoned_transform_test_index=-3,
    poisoned_target_transform_index=0,
    seed=global_random_seed,
    deterministic=True
)

# 实验根目录
exp_root_dir = config["exp_root_dir"]
# 攻击配置
schedule = {
    'device': f'cuda:{gpu_id}',

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    # 优化器需要的
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180], # 在 150和180epoch时调整lr

    'epochs': 200,

    'log_iteration_interval': 100, # 每过100个batch,记录下日志
    'test_epoch_interval': 10, # 每经过10个epoch,去测试下model效果
    'save_epoch_interval': 10, # 每经过10个epoch,保存下训练的model ckpt

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}

def attack():
    print("wanet后门攻击训练开始")
    wanet.train(schedule)
    # 后门模型
    backdoor_weight = torch.load(os.path.join( wanet.work_dir, "best_model.pth"), map_location="cpu")
    model.load_state_dict(backdoor_weight)
    # poisoned trainset
    poisoned_trainset = wanet.poisoned_train_dataset
    # poisoned_ids
    poisoned_ids = poisoned_trainset.poisoned_set
    
    save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    backdoor_data = {}
    backdoor_data["backdoor_model"] = model
    backdoor_data["poisoned_ids"]=poisoned_ids
    # trigger
    backdoor_data["identity_grid"]=identity_grid
    backdoor_data["noise_grid"]=noise_grid
    torch.save(backdoor_data, save_path)
    print(f"WaNet攻击完成,数据被存入{save_path}")

if __name__ == "__main__":
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    # 开始攻击并保存攻击模型和数据
    attack()

