'''
OurMethod主程序
'''

import sys
from utils.common_utils import my_excepthook
sys.excepthook = my_excepthook
from utils.common_utils import get_formattedDateTime
import os
from utils.common_utils import convert_to_hms
import time
from collections import Counter
import torch
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR
import setproctitle
from torch.utils.data import DataLoader,ConcatDataset,random_split
import torch.nn as nn
import torch.optim as optim
from utils.model_eval_utils import eval_asr_acc
from datasets.posisoned_dataset import get_all_dataset
from utils.common_utils import read_yaml,get_logger,set_random_seed
from utils.dataset_utils import get_class_num
from mid_data_loader import get_backdoor_data, get_class_rank
from defense.our.sample_select import clean_seed
from models.model_loader import get_model
from defense.our.sample_select import chose_retrain_set


def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          logger=None,lr_scheduler=None, class_weight = None, weight_decay=None):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    if lr_scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,T_max=num_epoch,eta_min=1e-6)
    if class_weight is None:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss(class_weight.to(device))
    loss_function.to(device)
    optimal_loss = float('inf')
    best_model = model
    for epoch in range(num_epoch):
        step_loss_list = []
        for _, batch in enumerate(dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
            step_loss_list.append(loss.item())
        if lr_scheduler:
            scheduler.step()
        epoch_loss = sum(step_loss_list) / len(step_loss_list)
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            best_model = model
        logger.info(f"epoch:{epoch},loss:{epoch_loss}")
    return model,best_model

def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model

def freeze_model(model,dataset_name,model_name):
    if dataset_name == "CIFAR10" or dataset_name == "GTSRB":
        if model_name == "ResNet18":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'features.5' in name or 'features.4' in name or 'features.3' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'dense4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    elif dataset_name == "ImageNet2012_subset":
        if model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name,param in model.named_parameters():
                if 'classifier' in name or 'features.denseblock4' in name or 'features.denseblock3' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "ResNet18":
            for name,param in model.named_parameters():
                if 'fc' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    else:
        raise Exception("模型不存在")
    return model

def get_class_weights(dataset,class_num):
    label_counts = Counter()
    for sample_id in range(len(dataset)):
        label = dataset[sample_id][1]
        label_counts[label] += 1
    most_num = label_counts.most_common(1)[0][1]
    weights = []
    for cls in range(class_num):
        num = label_counts[cls]
        weights.append(round(most_num / num,1))
    return label_counts, weights

def get_exp_info():
    # 获得实验时间戳: 年月日时分秒
    _time = get_formattedDateTime()
    exp_dir = os.path.join(exp_root_dir,"CleanSeedWithPoison",dataset_name,model_name,attack_name,f"exp_{r_seed}")
    os.makedirs(exp_dir,exist_ok=True)
    exp_info = {}
    exp_info["exp_time"] = _time
    exp_info["exp_name"] = "CleanSeedWithPoison"
    exp_info["exp_dir"] = exp_dir
    return exp_info
    
def pre_work(dataset_name, model_name, attack_name, r_seed):
    set_random_seed(r_seed)
    exp_info = get_exp_info()
    # 进程名称
    proctitle = f'{exp_info["exp_name"]}|{dataset_name}|{model_name}|{attack_name}|{r_seed}'
    setproctitle.setproctitle(proctitle)
    # 获得实验logger
    log_dir = os.path.join("log","CleanSeedWithPoison",dataset_name,model_name,attack_name,f"exp_{r_seed}")
    log_file_name  = f'{exp_info["exp_name"]}.log'
    logger = get_logger(log_dir,log_file_name)
    logger.info(f'实验时间:{exp_info["exp_time"]}')
    logger.info(f'实验名称:{exp_info["exp_name"]}')
    logger.info(f'实验目录:{exp_info["exp_dir"]}')
    logger.info(f"随机种子:{r_seed}")
    logger.info(f'进程title:{proctitle}')
    return logger,exp_info

def eval_and_save(model, filtered_poisoned_testset, clean_testset, device, save_path):
    asr, acc = eval_asr_acc(model,filtered_poisoned_testset,clean_testset,device)
    torch.save(model.state_dict(), save_path)
    return asr,acc

def one_scene(dataset_name, model_name, attack_name, r_seed):
    '''
    一个场景(dataset/model/attack)下的defense train
    '''
    # 实验开始计时
    start_time = time.perf_counter()
    # 实验开始前先设置好实验的一些元信息
    logger,exp_info= pre_work(dataset_name, model_name, attack_name, r_seed)

    # 加载后门攻击配套数据
    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    # 获得设备
    device = torch.device(f"cuda:{gpu_id}")
    # 空白模型
    blank_model = get_model(dataset_name, model_name) # 得到该数据集的模型
    seedSet,seed_id_list = clean_seed(poisoned_trainset, poisoned_ids, strict_clean=False) # 从中毒训练集中每个class选择10个clean seed
    seed_p_id_set = set(seed_id_list) & set(poisoned_ids)
    logger.info(f"种子中包含的中毒样本数量: {len(seed_p_id_set)}")
    # 种子微调原始的后门模型，为了保证模型的performance所以需要将后门模型进行部分层的冻结
    freeze_model(backdoor_model,dataset_name=dataset_name,model_name=model_name)
    # 种子微调训练30个轮次
    last_fine_tuned_model, best_fine_tuned_mmodel = train(backdoor_model,device,seedSet,num_epoch=30,lr=1e-3,logger=logger)
    # 把在种子训练集上表现最好的那个model保存下来，记为 best_BD_model.pth
    save_path = os.path.join(exp_info["exp_dir"], "best_BD_model.pth")
    asr,acc = eval_and_save(best_fine_tuned_mmodel, filtered_poisoned_testset, clean_testset, device, save_path)
    logger.info(f"best_fine_tuned_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
    save_path = os.path.join(exp_info["exp_dir"], "last_BD_model.pth")
    asr,acc = eval_and_save(last_fine_tuned_model, filtered_poisoned_testset, clean_testset, device, save_path)
    logger.info(f"last_fine_tuned_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")

    # 样本选择
    choice_rate = 0.6 # 打算选择60%的样本进行retrain
    class_rank = get_class_rank(dataset_name,model_name,attack_name) # 加载 class rank
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = chose_retrain_set(
        best_fine_tuned_mmodel, device, choice_rate, poisoned_trainset, poisoned_ids, class_rank=class_rank)
    logger.info(f"截取阈值:{choice_rate},中毒样本含量:{PN}/{len(choicedSet)}")
    # 防御重训练. 种子样本+选择的样本对模型进进下一步的 train
    availableSet = ConcatDataset([seedSet,choicedSet])
    epoch_num = 100 # 这次训练100个epoch
    lr = 1e-3
    batch_size = 512
    weight_decay=1e-3
    class_num = get_class_num(dataset_name)
    # 根据数据集中不同 class 的样本数量，设定不同 class 的 weight
    label_counter,weights = get_class_weights(availableSet, class_num)
    logger.info(f"label_counter:{label_counter}")
    logger.info(f"class_weights:{weights}")
    class_weights = torch.FloatTensor(weights)
    # 开始train,并返回最后一个epoch的model和在训练集上loss最小的那个best model
    last_defense_model,best_defense_model = train(
        best_fine_tuned_mmodel,device,availableSet,num_epoch=epoch_num,
        lr=lr, batch_size=batch_size, logger=logger, 
        lr_scheduler="CosineAnnealingLR",
        class_weight=class_weights,weight_decay=weight_decay)
    save_path = os.path.join(exp_info["exp_dir"], "best_defense_model.pth")
    asr,acc = eval_and_save(best_defense_model, filtered_poisoned_testset, clean_testset, device, save_path)
    logger.info(f"best_defense_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
    save_path = os.path.join(exp_info["exp_dir"], "last_defense_model.pth")
    asr,acc = eval_and_save(last_defense_model, filtered_poisoned_testset, clean_testset, device, save_path)
    logger.info(f"last_defense_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")

    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    logger.info(f"共耗时:{hours}时{minutes}分{seconds:.3f}秒")

if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name= "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
    model_name= "ResNet18" # ResNet18, VGG19, DenseNet
    attack_name ="IAD" # BadNets, IAD, Refool, WaNet, LabelConsistent
    gpu_id = 0
    r_seed = 1
    one_scene(dataset_name, model_name, attack_name, r_seed=r_seed)