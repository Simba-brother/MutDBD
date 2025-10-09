import sys
sys.path.append("./")
import os.path as osp
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize, Normalize,RandomResizedCrop,CenterCrop
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
from core import Refool
import setproctitle
from torchvision.models import vgg19
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset
from codes import config
from tqdm import tqdm

global_seed = 666
deterministic = False
torch.manual_seed(global_seed)

def _seed_worker():
    worker_seed =666
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# target model
model = vgg19(pretrained = True)
# 冻结预训练模型中所有参数的梯度
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后一个全连接层的输出类别数量
num_classes = 30  # 假设我们要改变分类数量为30
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)
# 存储反射照片
reflection_images = []
# URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" # "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set

def read_image(img_path, type=None):
    '''
    读取图片
    '''
    img = cv2.imread(img_path) # nd_array
    # cv2.imshow('Image', img)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError
# reflection image dir下所有的img path
reflection_image_path = os.listdir(reflection_data_dir)
# 读出来前200个reflection img
reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
# 训练集transform
transform_train = Compose([
    ToPILImage(), 
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(), # CHW
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ])
])
# 测试集transform
transform_test = Compose([
    ToPILImage(), 
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean = [ 0.485, 0.456, 0.406 ],
            std = [ 0.229, 0.224, 0.225 ]),
])
dataset_dir = "/data/mml/backdoor_detect/dataset/ImageNet2012_subset"

trainset = DatasetFolder(
    root=os.path.join(dataset_dir, "train"),
    loader=cv2.imread, # ndarray (H,W,C)
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root=os.path.join(dataset_dir, "val"),
    loader=cv2.imread, # ndarray(shape:HWC)
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)

# Configure the attack scheme
refool= Refool(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.1,
    poisoned_transform_train_index=1,
    poisoned_transform_test_index=1,
    poisoned_target_transform_index=1,
    schedule=None,
    seed=global_seed, # 666
    deterministic=deterministic, # True
    reflection_candidates = reflection_images, # reflection img list
)
exp_root_dir = config.exp_root_dir
dataset_name = "ImageNet"
model_name = "VGG19"
attack_name = "Refool"
schedule = {
    'device': 'cuda:1',
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

    'save_dir': osp.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
    'experiment_name': 'attack'
}


def attack():
    print("开始attack train")
    refool.train(schedule)
    print("attack train结束")
    work_dir = refool.work_dir
    poisoned_trainset = refool.poisoned_train_dataset
    poisoned_testset = refool.poisoned_test_dataset
    poisoned_ids = poisoned_trainset.poisoned_set
    poisonedTrainset = ExtractDataset(poisoned_trainset)
    poisonedTestset = ExtractDataset(poisoned_testset)
    pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
    purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)

    dict_state = {}
    dict_state["poisoned_trainset"]=poisonedTrainset
    dict_state["poisoned_ids"]=poisoned_ids
    dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
    dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
    dict_state["clean_testset"]=testset
    dict_state["poisoned_testset"]=poisonedTestset


    # 获得backdoor model weights
    backdoor_weights = torch.load(osp.join(work_dir, "best_model.pth"), map_location="cpu")
    # backdoor model存入字典数据中
    model.load_state_dict(backdoor_weights)
    dict_state["backdoor_model"] = model
    save_file_name = "dict_state.pth"
    save_path = osp.join(work_dir, save_file_name)
    print("开始保存攻击后数据")
    torch.save(dict_state, save_path)
    print(f"Refool攻击完成,数据和日志被存入{save_path}")

def eval(model,testset):
    '''
    评估接口
    '''
    model.eval()
    device = torch.device("cuda:0")
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
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    # backdoor_model
    backdoor_model = dict_state["backdoor_model"]

    poisoned_trainset = dict_state["poisoned_trainset"]
    poisoned_testset = dict_state["poisoned_testset"]
    clean_testset = dict_state["clean_testset"]
    purePoisonedTrainDataset = dict_state["purePoisonedTrainDataset"]
    pureCleanTrainDataset = dict_state["pureCleanTrainDataset"]
    assert len(pureCleanTrainDataset)+len(purePoisonedTrainDataset) == len(poisoned_trainset), "数量不对"

    poisoned_trainset_acc = eval(backdoor_model,poisoned_trainset)
    poisoned_testset_acc = eval(backdoor_model,poisoned_testset)
    clean_testset_acc = eval(backdoor_model,clean_testset)
    purePoisonedTrainDataset_acc = eval(backdoor_model,purePoisonedTrainDataset)
    pureCleanTrainDataset_acc = eval(backdoor_model,pureCleanTrainDataset)
    
    
    print("poisoned_trainset_acc",poisoned_trainset_acc)
    print("poisoned_testset_acc",poisoned_testset_acc)
    print("clean_testset_acc",clean_testset_acc)
    print("purePoisonedTrainDataset_acc",purePoisonedTrainDataset_acc)
    print("pureCleanTrainDataset_acc",pureCleanTrainDataset_acc)
    
    print("process_eval() success")
    
def get_dict_state():
    dict_state_file_path = os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name, "attack", "dict_state.pth")
    dict_state = torch.load(dict_state_file_path, map_location="cpu")
    return dict_state

def update_dict_state():
    pass

if __name__ == "__main__":
    setproctitle.setproctitle(dataset_name+"_"+model_name+"_"+attack_name+"_"+"attack")
    attack()
    # process_eval()
    # get_dict_state()
    # update_dict_state()

    pass