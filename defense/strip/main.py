

import os
import time
import random
from collections import Counter

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CosineAnnealingLR
from torch.utils.data import DataLoader,Subset,ConcatDataset,random_split

from models.model_loader import get_model
from mid_data_loader import get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from defense.our.sample_select import clean_seed
from utils.common_utils import convert_to_hms,set_random_seed
from datasets.utils import ExtractDataset_NoPoisonedFlag
from utils.dataset_utils import get_class_num
from utils.model_eval_utils import eval_asr_acc
from utils.save_utils import atomic_json_dump, load_results
from utils.common_utils import get_formattedDateTime


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

def eval_and_save(model, filtered_poisoned_testset, clean_testset, device, save_path):
    asr, acc = eval_asr_acc(model,filtered_poisoned_testset,clean_testset,device)
    torch.save(model.state_dict(), save_path)
    return asr,acc

def calc_entropy(_output: torch.Tensor) -> torch.Tensor:
        p = torch.nn.Softmax(dim=1)(_output) + 1e-8
        return (-p * p.log()).sum(1)

def superimpose(_input1: torch.Tensor, _input2: torch.Tensor, alpha: float = 0.5):
        result = _input1 + alpha * _input2
        return result

def check(model:nn.Module,_input: torch.Tensor, source_set:Dataset, device, N:int) -> torch.Tensor:
        # 存储该batch中每个sample与N个干净样本扰动下的的entropy
        _list = []
        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:N] # N 个干净数据集
        with torch.no_grad():
            for i in samples:
                x, y, isP = source_set[i]
                x = x.to(device)
                # batch 与X的扰动
                _test = superimpose(_input, x)
                _output = model(_test)
                # 该batch在X扰动下的entrop
                entropy = calc_entropy(_output).cpu().detach()
                _list.append(entropy)

        return torch.stack(_list).mean(0) # 该batch的entropy


def cleanse_cutoff(model:nn.Module,poisoned_dataset:Dataset,clean_dataset:Dataset, cut_off:float, device):
    # now cleanse the poisoned dataset with the chosen boundary
    poisoned_dataset_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False)
    # 所有样本的熵, 熵越小越可疑
    all_entropy = []
    for i, (_input, _label, _isP) in enumerate(poisoned_dataset_loader):
        _input = _input.to(device)
        entropies = check(model,_input, clean_dataset, device, N=100)
        for e in entropies:
            all_entropy.append(e.item())
    ranked_ids = np.argsort(all_entropy) # 熵越小的idx排越前
    cut = int(len(ranked_ids)*cut_off)
    suspicious_indices = ranked_ids[:cut]
    return all_entropy,suspicious_indices

    
    

def cleanse(model:nn.Module,poisoned_dataset:Dataset, clean_dataset:Dataset,device,defense_fpr:float=0.1):
    clean_entropy = []
    clean_set_loader = DataLoader(clean_dataset, batch_size=128, shuffle=False)
    # 按照批次遍历clean set
    for i, (_input, _label, _isP) in enumerate(clean_set_loader):
        _input = _input.to(device)
        entropies = check(model,_input, clean_dataset, device, N=100)
        for e in entropies:
            clean_entropy.append(e)
    clean_entropy = torch.FloatTensor(clean_entropy)

    clean_entropy, _ = clean_entropy.sort()
    threshold_low = float(clean_entropy[int(defense_fpr * len(clean_entropy))])
    threshold_high = np.inf

    # now cleanse the poisoned dataset with the chosen boundary
    poisoned_dataset_loader = DataLoader(poisoned_dataset, batch_size=128, shuffle=False)
    # 所有样本的熵
    all_entropy = []
    for i, (_input, _label, _isP) in enumerate(poisoned_dataset_loader):
        _input = _input.to(device)
        entropies = check(model,_input, clean_dataset, device, N=100)
        for e in entropies:
            all_entropy.append(e)
    all_entropy = torch.FloatTensor(all_entropy)
    suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
    return all_entropy,suspicious_indices


def train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          lr_scheduler=None, class_weight = None, weight_decay=None, early_stop=False):
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
    patience = 5
    count = 0
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
        print(f"epoch:{epoch},loss:{epoch_loss}")
        if epoch_loss < optimal_loss:
            count = 0
            optimal_loss = epoch_loss
            best_model = copy.deepcopy(model)
        else:
            count += 1
            if early_stop and count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return model,best_model

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


def get_backdoor_base_data(dataset_name, model_name, attack_name):
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
    return backdoor_model,poisoned_ids, poisoned_trainset,filtered_poisoned_testset, clean_trainset, clean_testset


def sample_select(clean_seedSet,poisoned_trainset,backdoor_model,hard_cut_flag:bool = True):
    backdoor_model.eval()
    backdoor_model.to(device)
    if hard_cut_flag:
        all_entropy,suspicious_indices = cleanse_cutoff(backdoor_model,poisoned_trainset,clean_seedSet,
                                                        0.4,device)
    else:
        all_entropy,suspicious_indices = cleanse(backdoor_model, poisoned_trainset, clean_seedSet,device,defense_fpr=0.1)
    return all_entropy,suspicious_indices

def one_scene(dataset_name, model_name, attack_name,save_dir):

    # 先得到后门攻击基础数据
    backdoor_model,poisoned_ids, poisoned_trainset,filtered_poisoned_testset, clean_trainset, clean_testset = \
        get_backdoor_base_data(dataset_name, model_name, attack_name)

    # select samples
    sample_select_start_time = time.perf_counter()
    clean_seedSet, _ = clean_seed(poisoned_trainset,poisoned_ids,strict_clean=True)
    all_entropy,suspicious_indices = sample_select(clean_seedSet,poisoned_trainset,backdoor_model,
                                                   hard_cut_flag)
    sample_select_end_time = time.perf_counter()
    sample_select_cost_time = sample_select_end_time - sample_select_start_time
    hours, minutes, seconds = convert_to_hms(sample_select_cost_time)
    print(f"select samples耗时:{hours}时{minutes}分{seconds:.1f}秒")

    if save_entropy:
        save_path = os.path.join(save_dir,"entropy.pt")
        torch.save(all_entropy,save_path)
        print(f"训练集中所有样本的熵保存在: {save_path}")
        save_path = os.path.join(save_dir,"suspicious_indices.pt")
        torch.save(suspicious_indices,save_path)
        print(f"训练集中可疑的样本索引保存在: {save_path}")

    suspicious_ids = suspicious_indices.tolist()
    all_ids = list(range(len(poisoned_trainset)))
    remain_ids = list(set(all_ids) - set(suspicious_ids))
    PN = len(list(set(remain_ids) & set(poisoned_ids)))
    print(f"准备用于retrain的数据集{PN}/{len(remain_ids)}")

    # defense retrain
    retrain_start_time = time.perf_counter()
    ranker_model_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours",
                                                dataset_name,model_name,attack_name,
                                                f"exp_{r_seed}","best_BD_model.pth")
    ranker_model_state_dict = torch.load(ranker_model_state_dict_path,map_location="cpu")
    model = get_model(dataset_name, model_name)
    model.load_state_dict(ranker_model_state_dict)
    ranker_model = model
    ranker_asr, ranker_acc = eval_asr_acc(ranker_model,filtered_poisoned_testset,clean_testset,device)
    print(f"ranker_asr:{ranker_asr},ranker_acc:{ranker_acc}")
    choicedSet = Subset(poisoned_trainset,remain_ids)
    # 防御重训练. 种子样本+选择的样本对模型进进下一步的 train
    availableSet = ConcatDataset([clean_seedSet,choicedSet])
    class_num = get_class_num(dataset_name)
    epoch_num = 100 # 这次训练100个epoch
    lr = 1e-3
    batch_size = 512
    weight_decay=1e-3
    # 根据数据集中不同 class 的样本数量，设定不同 class 的 weight
    label_counter,weights = get_class_weights(availableSet, class_num)
    print(f"label_counter:{label_counter}")
    print(f"class_weights:{weights}")
    class_weights = torch.FloatTensor(weights)
    # 开始train,并返回最后一个epoch的model和在训练集上loss最小的那个best model

    last_defense_model,best_defense_model = train(
        ranker_model,device,availableSet,num_epoch=epoch_num,
        lr=lr, batch_size=batch_size,
        lr_scheduler="CosineAnnealingLR",
        class_weight=class_weights,weight_decay=weight_decay,early_stop=True)
    retrain_end_time = time.perf_counter()
    retrain_cost_time = retrain_end_time - retrain_start_time
    hours, minutes, seconds = convert_to_hms(retrain_cost_time)
    print(f"retrain耗时:{hours}时{minutes}分{seconds:.1f}秒")

    if save_model:
        # 保存 defense retrain 结果
        best_save_path = os.path.join(save_dir, "best_defense_model.pth")
        best_asr,best_acc = eval_and_save(best_defense_model, filtered_poisoned_testset, clean_testset, device, 
                                          best_save_path)
        print(f"best_defense_model|ASR:{best_asr},ACC:{best_acc},权重保存在:{best_save_path}")
        last_save_path = os.path.join(save_dir, "last_defense_model.pth")
        last_asr,last_acc = eval_and_save(last_defense_model, filtered_poisoned_testset, clean_testset, device, 
                                          last_save_path)
        print(f"last_defense_model|ASR:{last_asr},ACC:{last_acc},权重保存在:{last_save_path}")
    else:
        best_asr, best_acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
        last_asr, last_acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
    res = {
        "PN":PN,
        "best_asr":best_asr,
        "best_acc":best_acc,
        "last_asr":last_asr,
        "last_acc":last_acc,
    }
    print(f"PN:{PN},best_asr:{best_asr},best_acc:{best_acc},last_asr:{last_asr},last_acc:{last_acc}")
    return res


def save_experiment_result(exp_save_path, 
                           dataset_name, model_name, attack_name,r_seed,
                           result_data
                          ):
    """
    保存单个实验结果到嵌套JSON
    结构: {dataset: {model: {attack: {beta: {r_seed: result}}}}}
    """
    # 加载现有数据
    data = load_results(exp_save_path)

    # 构建嵌套结构
    if dataset_name not in data:
        data[dataset_name] = {}
    if model_name not in data[dataset_name]:
        data[dataset_name][model_name] = {}
    if attack_name not in data[dataset_name][model_name]:
        data[dataset_name][model_name][attack_name] = {}

    # 保存结果
    data[dataset_name][model_name][attack_name][str(r_seed)] = result_data

    # 原子写入
    atomic_json_dump(data, exp_save_path)

if __name__ == "__main__":
    # one-scence
    '''
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name= "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
    model_name= "ResNet18" # ResNet18, VGG19, DenseNet
    attack_name ="BadNets" # BadNets, IAD, Refool, WaNet, LabelConsistent
    gpu_id = 1
    r_seed = 1
    
    device = torch.device(f"cuda:{gpu_id}")
    start_time = time.perf_counter()
    print("==="*30)
    print(f"Strip|sample select|{dataset_name}|{model_name}|{attack_name}|r_seed:{r_seed}")
    set_random_seed(r_seed)
    save_dir = os.path.join(exp_root_dir,"Defense","Strip",dataset_name,model_name,attack_name)
    one_scene(dataset_name, model_name, attack_name, save_dir)
    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
    '''

    # all scence
    cur_pid = os.getpid()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    
    exp_name = "Strip_hardCut"
    exp_time = get_formattedDateTime()
    exp_save_dir = os.path.join(exp_root_dir,"Defense",exp_name)
    os.makedirs(exp_save_dir,exist_ok=True)
    exp_save_file_name = "results.json"
    exp_save_path = os.path.join(exp_save_dir,exp_save_file_name)
    save_model = False
    save_entropy = True

    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))
    hard_cut_flag = True 
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}")

    print("PID:",cur_pid)
    print("exp_root_dir:",exp_root_dir)
    print("exp_name:",exp_name)
    print("exp_time:",exp_time)
    print("exp_save_path:",exp_save_path)
    print("save_model:",save_model)
    print("dataset_name_list:",dataset_name_list)
    print("model_name_list:",model_name_list)
    print("attack_name_list:",attack_name_list)
    print("r_seed_list:",r_seed_list)
    print("gpu_id:",gpu_id)
    print("hard_cut_flag:",hard_cut_flag)

    all_start_time = time.perf_counter()
    for r_seed in r_seed_list:
        one_repeat_start_time = time.perf_counter()
        set_random_seed(r_seed)
        for dataset_name in dataset_name_list:
            for model_name in model_name_list:
                for attack_name in attack_name_list:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    one_sence_start_time = time.perf_counter()
                    print(f"\n{exp_name}|{dataset_name}|{model_name}|{attack_name}|r_seed={r_seed}|time={get_formattedDateTime()}")
                    if save_model or save_entropy:
                        save_dir = os.path.join(exp_save_dir,dataset_name,model_name,attack_name)
                        os.makedirs(save_dir,exist_ok=True)
                    else:
                        save_dir = None
                    res = one_scene(dataset_name, model_name, attack_name, save_dir)
                    save_experiment_result(exp_save_path, 
                           dataset_name, model_name, attack_name,r_seed,
                           res)
                    one_scence_end_time = time.perf_counter()
                    one_scence_cost_time = one_scence_end_time - one_sence_start_time
                    hours, minutes, seconds = convert_to_hms(one_scence_cost_time)
                    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
        one_repeat_end_time = time.perf_counter()
        one_repeart_cost_time = one_repeat_end_time - one_repeat_start_time
        hours, minutes, seconds = convert_to_hms(one_repeart_cost_time)
        print(f"\n一轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
    all_end_time = time.perf_counter()
    all_cost_time = all_end_time - all_start_time
    hours, minutes, seconds = convert_to_hms(all_cost_time)
    print(f"\n{len(r_seed_list)}轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
