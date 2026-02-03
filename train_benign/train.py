import os
import time
import torch
import random
from datasets.clean_dataset import get_clean_dataset
from models.model_loader import get_model
from trainer import NeuralNetworkTrainer
from utils.model_eval_utils import EvalModel
from torch.utils.data import DataLoader,Subset
from utils.common_utils import convert_to_hms,get_formattedDateTime

def get_data_loder(dataset,shuffle,
                   batch_size:int = 128,
                   num_workers:int = 8,
                   drop_last:bool = False,
                   pin_memory:bool = True
                   ):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
            )


def split_val(dataset, split_rate:float=0.1):
    all_id_list = list(range(len(dataset)))

    val_id_list = random.sample(all_id_list, int(len(all_id_list)*split_rate))
    train_id_list = [id for id in all_id_list if id not in val_id_list]

    train_dataset = Subset(dataset,train_id_list)
    val_dataset = Subset(dataset,val_id_list)
    return train_dataset, val_dataset

def main_one_scence(dataset_name,model_name,attack_name,save_dir):
    # 加载数据集
    alltrainset, testset = get_clean_dataset(dataset_name,attack_name)
    trainset, valset = split_val(alltrainset)
    trainset_loader = get_data_loder(trainset,shuffle=True)
    valset_loader = get_data_loder(valset,shuffle=False)
    # 加载模型
    model = get_model(dataset_name, model_name)
    device = torch.device(f"cuda:{gpu_id}")
    trainer = NeuralNetworkTrainer(model,device,init_lr = 0.001,model_dir=save_dir)
    # 训练模型
    history = trainer.fit(
        train_loader=trainset_loader,
        val_loader=valset_loader,
        epochs=100,
        early_stopping_patience=10,
        save_best=True
    )
    # 加载出在valset上loss最低的model
    model.load_state_dict(trainer.best_model_state)
    em = EvalModel(trainer.model,testset,device,batch_size=128)
    test_acc = em.eval_acc()
    print(f"Test acc:{test_acc}")

if __name__ == "__main__":

    pid = os.getpid()
    # 实验基本信息
    exp_name = "BenignTrain"
    exp_time = get_formattedDateTime()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_save_dir = os.path.join(exp_root_dir,exp_name)
    os.makedirs(exp_save_dir,exist_ok=True)

    print("PID:",pid)
    print("exp_name:",exp_name)
    print("exp_time:",exp_time)
    print("exp_save_dir:",exp_save_dir)

    # 实验参数
    r_seed = 42
    random.seed(r_seed)
    gpu_id = 1
    print("r_seed:",r_seed)
    print("gpu_id:",gpu_id)
    # 本脚本参数
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    print(f"{dataset_name}|{model_name}|{attack_name}")
    save_dir = os.path.join(exp_save_dir,dataset_name,model_name,attack_name)
    os.makedirs(save_dir,exist_ok=True)
    print("save_dir:",save_dir)
    start_time = time.perf_counter()
    main_one_scence(dataset_name,model_name,attack_name,save_dir)
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    print(f"耗时:{hours}时{minutes}分{seconds:.1f}秒")