import os
import torch
import torch.nn as nn
from attack.core.attacks import IAD
from attack.core.attacks.IAD import Generator 
import setproctitle
from commonUtils import read_yaml
from datasets.clean_dataset import get_clean_dataset
from attack.iad_utils import IADPoisonedDatasetFolder
from attack.random_util import set_random_seed
from attack.models import get_model

config = read_yaml("config.yaml")
global_random_seed = config["global_random_seed"]
set_random_seed(global_random_seed)

dataset_name = "GTSRB"
model_name = "ResNet18"
attack_name = "IAD"
gpu_id = 1
target_class = config["target_class"]
poisoned_rate = config["poisoned_rate"]

clean_trainset_1,clean_testset_1 = get_clean_dataset(dataset_name,attack_name)
clean_trainset_2,clean_testset_2 = get_clean_dataset(dataset_name,attack_name)
# victim model
model = get_model(dataset_name,model_name)
# 实验根目录
exp_root_dir = config["exp_root_dir"]
# 攻击配置
schedule = {
    'device': f'cuda:{gpu_id}',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'milestones': [100, 200, 300, 400],
    'lambda': 0.1,
    
    'lr_G': 0.01,
    'betas_G': (0.5, 0.9),
    'milestones_G': [200, 300, 400, 500],
    'lambda_G': 0.1,

    'lr_M': 0.01,
    'betas_M': (0.5, 0.9),
    'milestones_M': [10, 20],
    'lambda_M': 0.1,
    
    'epochs': 600,
    'epochs_M': 25,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name),
    'experiment_name': 'ATTACK'
}
# 攻击类
iad = IAD(
    dataset_name=dataset_name, # 不要变
    train_dataset=clean_trainset_1,
    test_dataset=clean_testset_1,
    train_dataset1=clean_trainset_2,
    test_dataset1=clean_testset_2,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=target_class,
    poisoned_rate=poisoned_rate,
    cross_rate=poisoned_rate,
    lambda_div=1,
    lambda_norm=100,
    mask_density=0.032,
    EPSILON=1e-7,
    schedule=schedule,
    seed=global_random_seed,
    deterministic=True
)

def attack():
    iad.train()
    dict_state = torch.load(os.path.join(iad.work_dir,"dict_state.pth"), map_location="cpu")
    model.load_state_dict(dict_state["model"])

    backdoor_model = model
    modelG = Generator(dataset_name)
    modelM = Generator(dataset_name, out_channels=1)
    
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])

    poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = clean_trainset_1,
        y_target = target_class,
        poisoned_rate = poisoned_rate,
        modelG = modelG,
        modelM =modelM
    )
    backdoor_model.eval()
    modelG.eval()
    modelM.eval()    
    poisoned_ids = poisoned_trainset.poisoned_set
    save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
    backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_ids":poisoned_ids,
        "modelG":modelG.state_dict(),
        "modelM":modelM.state_dict()
    }
    torch.save(backdoor_data,save_path)
    print(f"backdoor_data is saved in {save_path}")

if __name__ == "__main__":
    proc_title = "ATTACK|"+dataset_name+"|"+attack_name+"|"+model_name
    setproctitle.setproctitle(proc_title)
    print(proc_title)
    attack()

