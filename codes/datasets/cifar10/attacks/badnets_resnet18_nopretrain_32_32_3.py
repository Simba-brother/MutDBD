import sys
sys.path.append("./")
import os.path as osp
import joblib
import time
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

from codes.core.attacks import BadNets

from codes.core.models.resnet import ResNet
from codes.modelMutat import ModelMutat
from codes.eval_model import EvalModel
from codes import draw
from codes.utils import create_dir
from collections import defaultdict
from tqdm import tqdm

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 训练集transform    
transform_train = Compose([
    ToPILImage(),
    RandomHorizontalFlip(),
    ToTensor()
])
# 测试集transform
transform_test = Compose([
    ToPILImage(),
    ToTensor()
])

# target model
model = ResNet(num=18,num_classes=10)
# 获得数据集
trainset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='/data/mml/backdoor_detect/dataset/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

# backdoor pattern
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0


badnets = BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

class PureCleanTrainDataset(Dataset):
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.pureCleanTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        pureCleanTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label = self.poisoned_train_dataset[id]
            if id not in self.poisoned_ids:
                pureCleanTrainDataset.append((sample,label))
        return pureCleanTrainDataset
    
    def __len__(self):
        return len(self.pureCleanTrainDataset)
    
    def __getitem__(self, index):
        x,y=self.pureCleanTrainDataset[index]
        return x,y

class PurePoisonedTrainDataset(Dataset):
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.purePoisonedTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        purePoisonedTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label = self.poisoned_train_dataset[id]
            if id in self.poisoned_ids:
                purePoisonedTrainDataset.append((sample,label))
        return purePoisonedTrainDataset
    
    def __len__(self):
        return len(self.purePoisonedTrainDataset)
    
    def __getitem__(self, index):
        x,y=self.purePoisonedTrainDataset[index]
        return x,y
    
# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'cuda:1',
    
    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': '/data/mml/backdoor_detect/experiments',
    'experiment_name': 'cifar10_resnet18_nopretrain_32_32_3_badnets'
}


def attack():
    # 攻击
    badnets.train(schedule)
    # 工作dir
    work_dir = badnets.work_dir
    # 获得backdoor model weights
    backdoor_model = badnets.best_model
    # clean testset
    clean_testset = testset
    # poisoned testset
    poisoned_testset = badnets.poisoned_test_dataset
    # poisoned trainset
    poisoned_trainset = badnets.poisoned_train_dataset
    # poisoned_ids
    poisoned_ids = poisoned_trainset.poisoned_set
    # pure clean trainset
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    # pure poisoned trainset
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)

    dict_state = {}
    dict_state["backdoor_model"] = backdoor_model
    dict_state["poisoned_trainset"]=poisoned_trainset
    dict_state["poisoned_ids"]=poisoned_ids
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    dict_state["clean_testset"]=clean_testset
    dict_state["poisoned_testset"]=poisoned_testset
    dict_state["pattern"] = pattern
    dict_state['weight']=weight
    save_file_name = "dict_state.pth"
    save_path = osp.join(work_dir, save_file_name)
    torch.save(dict_state, save_path)
    print(f"BadNets攻击完成,数据和日志被存入{save_path}")

def eval(model,testset):
    model.eval()
    device = torch.device("cuda:5")
    model.to(device)
    batch_size = 128
    # 加载trigger set
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    # 测试集总数
    total_num = len(testset_loader.dataset)
    # 评估开始时间
    start = time.time()
    acc = torch.tensor(0., device=device)
    correct_num = 0 # 攻击成功数量
    with torch.no_grad():
        for batch_id, batch in enumerate(testset_loader):
            X = batch[0]
            Y = batch[1]
            X = X.to(device)
            Y = Y.to(device)
            pridict_digits = model(X)
            correct_num += (torch.argmax(pridict_digits, dim=1) == Y).sum()
        acc = correct_num / total_num
        acc = round(acc.item(),3)
    end = time.time()
    print("acc:",acc)
    print(f'Total eval() time: {end-start:.1f} seconds')
    return acc

def process_eval():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth",map_location="cpu")
    backdoor_model = dict_state["backdoor_model"]
    clean_testset = dict_state["clean_testset"]
    poisoned_testset = dict_state["poisoned_testset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    poisoned_trainset = dict_state["poisoned_trainset"]

    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    benign_testset_acc = eval(backdoor_model,clean_testset)
    poisoned_testset_acc = eval(backdoor_model, poisoned_testset)
    pure_clean_trainset_acc = eval(backdoor_model, pureCleanTrainDataset)
    pure_poisoned_trainset_acc = eval(backdoor_model, purePoisonedTrainDataset)
    print("clean_testset_acc", benign_testset_acc)
    print("poisoned_testset_acc", poisoned_testset_acc)
    print("pure_clean_trainset_acc", pure_clean_trainset_acc)
    print("pure_poisoned_trainset_acc", pure_poisoned_trainset_acc)

def get_dict_state():
    dict_state = torch.load("/data/mml/backdoor_detect/experiments/cifar10_resnet_nopretrain_32_32_3_badnets_2023-11-12_21:11:53/dict_state.pth", map_location="cpu")
    return dict_state


def gf_mutate():
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    scale = 5
    for mutation_ratio in mutation_ratio_list:
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/gf/ratio_{mutation_ratio}_scale_{scale}_num_{mutation_model_num}/{attack_name}"
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._gf_mut(scale)    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = osp.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")


def neuron_activation_inverse_mutate():
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "neuron_activation_inverse"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    for mutation_ratio in mutation_ratio_list:
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}"
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_activation_inverse()    
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = osp.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_block_mutate():
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "neuron_block"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    for mutation_ratio in mutation_ratio_list:
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}"
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_block()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = osp.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def neuron_switch_mutate():
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "neuron_switch"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    for mutation_ratio in mutation_ratio_list:
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}"
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._neuron_switch()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = osp.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")

def weight_shuffling():
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "weight_shuffle"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    for mutation_ratio in mutation_ratio_list:
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}"
        create_dir(work_dir)
        for m_i in range(mutation_model_num):
            muodel_mutat = ModelMutat(backdoor_model, mutation_ratio)
            mutated_model = muodel_mutat._weight_shuffling()   
            save_file_name = f"model_mutated_{m_i+1}.pth"
            save_path = osp.join(work_dir, save_file_name)
            torch.save(mutated_model.state_dict(), save_path)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}, save_path:{save_path}")


def eval_mutated_model():
    device = torch.device("cuda:2")
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "weight_shuffle"
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    mutation_model_num = 50
    base_dir = "/data/mml/backdoor_detect/experiments"
    scale = 5
    save_file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    save_path = osp.join(base_dir,save_file_name)
    data = defaultdict(list)
    for mutation_ratio in tqdm(mutation_ratio_list):
        work_dir = f"/data/mml/backdoor_detect/experiments/{dataset_name}/{model_name}/mutates/{mutation_name}/ratio_{mutation_ratio}_num_{mutation_model_num}/{attack_name}"        
        for m_i in range(mutation_model_num):
            state_dict = torch.load(osp.join(work_dir, f"model_mutated_{m_i+1}.pth"), map_location="cpu")
            backdoor_model.load_state_dict(state_dict)
            evalModel = EvalModel(backdoor_model, poisoned_trainset, device)
            report = evalModel._eval_classes_acc()
            data[mutation_ratio].append(report)
            print(f"mutation_ratio:{mutation_ratio}, m_i:{m_i}")
    joblib.dump(data, save_path)




def draw_box():
    device = torch.device("cuda:0")
    dict_state = get_dict_state()
    backdoor_model = dict_state["backdoor_model"]
    poisoned_trainset = dict_state["poisoned_trainset"]
    e = EvalModel(backdoor_model, poisoned_trainset, device)
    origin_report = e._eval_classes_acc()
    dataset_name = "CIFAR10"
    model_name = "resnet18_nopretrain_32_32_3"
    attack_name = "BadNets"
    mutation_name = "gf"
    base_dir = "/data/mml/backdoor_detect/experiments"
    file_name = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}.data"
    file_path = osp.join(base_dir,file_name)
    data = joblib.load(file_path)
    mutation_ratio_list = [0.01, 0.05, 0.1, 0.15, 0.20, 0.3, 0.4, 0.5, 0.6, 0.8]
    ans = {}
    for mutation_ratio in mutation_ratio_list:
        ans[mutation_ratio] = {}
        for class_idx in range(10):
            ans[mutation_ratio][class_idx] = []
    for mutation_ratio in mutation_ratio_list:
        report_list = data[mutation_ratio]
        for report in report_list:
            for class_idx in range(10):
                precision = report[str(class_idx)]["precision"]
                origin_precison = origin_report[str(class_idx)]["precision"]
                dif = round(origin_precison-precision,3)
                ans[mutation_ratio][class_idx].append(dif)

    save_dir = osp.join(base_dir, "images/box")
    create_dir(save_dir)
    for mutation_ratio in mutation_ratio_list:
        all_y = []
        labels = []
        for class_i in range(10):
            y_list = ans[mutation_ratio][class_i]
            all_y.append(y_list)
            labels.append(f"Class_{class_i}")
        title = f"{dataset_name}_{model_name}_{attack_name}_{mutation_name}_{mutation_ratio}"
        save_file_name = title+".png"
        save_path = osp.join(save_dir, save_file_name)
        draw.draw_box(all_y, labels,title,save_path)
            
if __name__ == "__main__":
    # attack()
    # process_eval()
    # get_dict_state()
    # gf_mutate()
    # neuron_activation_inverse_mutate()
    # neuron_block_mutate()
    # neuron_switch_mutate()
    # weight_shuffling()
    # eval_mutated_model()
    # draw_box()
    pass