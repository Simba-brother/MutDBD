from typing import Pattern
import sys
sys.path.append("./")
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip,ToPILImage
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
from codes import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
temp_patch = 0.5 * torch.ones(3, 8, 8)
patch = torch.bernoulli(temp_patch)


# trigger = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]])
# patch = trigger.repeat((3, 1, 1))

def show_dataset(dataset, num, path_to_save):
    """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    for i in range(num):
        ax = plt.subplot(num,1,i+1)
        img = (dataset[i][0]).permute(1,2,0).cpu().detach().numpy()
        ax.imshow(img)
    plt.savefig(path_to_save)

def init_model(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

torch.manual_seed(global_seed)

victim_model = core.models.ResNet(18,num_classes=10)

# Prepare datasets
transform_train = Compose([ # the data augmentation method is hard-coded in core.SleeperAgent, user-defined data augmentation is not allowed
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])
transform_test = Compose([
    ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor(),
])
trainset = DatasetFolder(
    root='./dataset/cifar10/train',
    loader=cv2.imread, # ndarray
    extensions=('png',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)
testset = DatasetFolder(
    root='./dataset/cifar10/test',
    loader=cv2.imread,
    extensions=('png',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


sleeper_agent = core.SleeperAgent(
    train_dataset=trainset,
    test_dataset=testset,
    model=victim_model,
    loss=nn.CrossEntropyLoss(),
    patch=patch,
    random_patch=True,
    eps=16./255,
    y_target=1,
    y_source=2,    
    poisoned_rate=0.01,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic,
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
    
schedule = {
    'device': 'cuda:4',
    # 'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 75],
    
    # 'pretrain': 'pretrain/pretrain_cifar.pth',

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'cifar10_resnet_nopretrained_32_32_3_SleeperAgent',


    'pretrain_schedule': {'epochs':100, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[50,75], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'retrain_schedule': {'epochs':40, 'lr':0.1, 'weight_decay': 5e-4,  'gamma':0.1, 'milestones':[14,24,35], 'batch_size':128, 'num_workers':8, 'momentum': 0.9,},
    'craft_iters': 250, # total iterations to craft the poisoned trainset
    'retrain_iter_interval': 50, # retrain the model after #retrain_iter_interval crafting iterations
    # retrain_iter_interval, 经过50轮次 更新 delta 后 要更新一下 model 为了更好地得到 delta
    # milestones for retrain: [epochs // 2.667, epochs // 1.6, epochs // 1.142]
}

def attack():
    print("sleeperAgent开始攻击")
    sleeper_agent.train(init_model, schedule)
    backdoor_model = sleeper_agent.best_model
    work_dir = sleeper_agent.work_dir
    clean_testset = testset
    poisoned_trainset, poisoned_testset = sleeper_agent.get_poisoned_dataset()
    poison_ids= sleeper_agent.sleeper_agent
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poison_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poison_ids)
    print("sleeperAgent攻击结束,开始保存攻击数据")
    dict_state = {}
    dict_state["backdoor_model"] = backdoor_model
    dict_state["clean testset"] = clean_testset
    dict_state["poisoned_testset"] = poisoned_testset
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    torch.save(dict_state, os.path.join(work_dir, "dict_state.pth"))
    print(f"攻击数据被保存到:{os.path.join(work_dir, 'dict_state.pth')}")
    print("attack() finished")

if __name__ == "__main__":
    attack()
    #show_dataset(poisoned_trainset, 5, 'cifar_train_poison.png')
    #show_dataset(poisoned_testset, 5, 'cifar_test_poison.png')

