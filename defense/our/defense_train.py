'''
OurMethod主程序
'''

import sys
import json
import copy
from utils.common_utils import my_excepthook
sys.excepthook = my_excepthook
from utils.common_utils import get_formattedDateTime
import os
from utils.common_utils import convert_to_hms
import time
from collections import Counter
import torch
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR

import setproctitle
from torch.utils.data import DataLoader,ConcatDataset
import torch.nn as nn
import torch.optim as optim
from utils.model_eval_utils import eval_asr_acc
from datasets.posisoned_dataset import get_all_dataset
from datasets.clean_dataset import get_cifar10_trans_cleanseed_dataset
from utils.common_utils import set_random_seed
from utils.dataset_utils import get_class_num
from mid_data_loader import get_backdoor_data, get_class_rank
from defense.our.sample_select import clean_seed,get_test_clean_seed
from models.model_loader import get_model
from defense.our.sample_select import chose_retrain_set
from defense.our.semi_train_utils import semi_train
from utils.save_utils import atomic_json_dump, load_results




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



def eval_and_save(model, filtered_poisoned_testset, clean_testset, device, save_path):
    asr, acc = eval_asr_acc(model,filtered_poisoned_testset,clean_testset,device)
    torch.save(model.state_dict(), save_path)
    return asr,acc

def one_scene(dataset_name, model_name, attack_name, r_seed, save_dir=None):
    '''
    一个场景(dataset/model/attack)下的defense train
    '''
    # 实验开始计时
    start_time = time.perf_counter()

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

    if test_seed is True:
        seedSet = get_test_clean_seed(clean_testset)
    elif trans_seed is True:
        seedSet = get_cifar10_trans_cleanseed_dataset(attack_name)
    else:
        seedSet,seed_id_list = clean_seed(poisoned_trainset, poisoned_ids,
                                      strict_clean=strict_clean,seed_poisoned_num=seed_poisoned_num) # 从中毒训练集中每个class选择10个clean seed
        seed_p_id_set = set(seed_id_list) & set(poisoned_ids)
        print(f"种子中包含的中毒样本数量: {len(seed_p_id_set)}/{len(seed_id_list)}")

        
    if resume_ranker_model is False:
        
        if freeze_model_flag:
            freeze_model(backdoor_model,dataset_name=dataset_name,model_name=model_name)
        # 种子微调训练30个轮次
        seed_finetune_init_lr = 1e-3 # 1e-3
        seed_finetune_epochs = 100 # 30
        last_fine_tuned_model, best_fine_tuned_mmodel = train(backdoor_model,device,seedSet,
                                                              num_epoch=seed_finetune_epochs,
                                                              lr=seed_finetune_init_lr,early_stop=False)
        if save_model:
            # 把在种子训练集上表现最好的那个model保存下来，记为 best_BD_model.pth
            save_path = os.path.join(save_dir, "best_BD_model.pth")
            asr,acc = eval_and_save(best_fine_tuned_mmodel, filtered_poisoned_testset, clean_testset, device, save_path)
            print(f"best_fine_tuned_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
            save_path = os.path.join(save_dir, "last_BD_model.pth")
            asr,acc = eval_and_save(last_fine_tuned_model, filtered_poisoned_testset, clean_testset, device, save_path)
            print(f"last_fine_tuned_model|ASR:{asr},ACC:{acc},权重保存在:{save_path}")
        else:
            best_finetune_asr, best_finetune_acc = eval_asr_acc(best_fine_tuned_mmodel,filtered_poisoned_testset,clean_testset,device)
            last_finetune_asr, last_finetune_acc = eval_asr_acc(last_fine_tuned_model,filtered_poisoned_testset,clean_testset,device)
            print(f"best_fine_tuned_model|ASR:{best_finetune_asr},ACC:{best_finetune_acc}")
            print(f"last_fine_tuned_model|ASR:{last_finetune_asr},ACC:{last_finetune_acc}")
        ranker_model = best_fine_tuned_mmodel
    else:
        ranker_model_state_dict_path = os.path.join(exp_root_dir,"Defense","Ours",dataset_name,model_name,attack_name,
                                                f"exp_{r_seed}","best_BD_model.pth")
        ranker_model_state_dict = torch.load(ranker_model_state_dict_path,map_location="cpu")
        model = get_model(dataset_name, model_name)
        model.load_state_dict(ranker_model_state_dict)
        ranker_model = model

    # 样本选择
    class_rank = get_class_rank(dataset_name,model_name,attack_name) # 加载 class rank
    choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = chose_retrain_set(
        ranker_model, device, choice_rate, poisoned_trainset, poisoned_ids, class_rank=class_rank,
        beta=beta,sigmoid_fag=sigmoid_fag)
    print(f"截取阈值:{choice_rate},中毒样本含量:{PN}/{len(choicedSet)}")
    class_num = get_class_num(dataset_name)
    if semi == True:
        epochs = 120
        lr = 2e-3
        all_id_list = list(range(len(poisoned_trainset)))
        labeled_id_set = set(seed_id_list) | set(choiced_sample_id_list) 
        unlabeled_id_set = set(all_id_list) - labeled_id_set
        last_defense_model,best_defense_model = semi_train(
            ranker_model,device,class_num,seedSet,epochs,lr,poisoned_trainset,
            labeled_id_set,unlabeled_id_set,all_id_list)

    else:
        # 防御重训练. 种子样本+选择的样本对模型进进下一步的 train
        if test_seed is True or trans_seed is True:
            availableSet = choicedSet
        else:
            availableSet = ConcatDataset([seedSet,choicedSet])
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
            class_weight=class_weights,weight_decay=weight_decay,early_stop=False)
    
    if save_model:
        best_save_path = os.path.join(save_dir, "best_defense_model.pth")
        best_asr,best_acc = eval_and_save(best_defense_model, filtered_poisoned_testset, clean_testset, device, 
                                best_save_path)
        print(f"best_defense_model|ASR:{best_asr},ACC:{best_acc},权重保存在:{save_path}")
        last_save_path = os.path.join(save_dir, "last_defense_model.pth")
        best_asr,best_acc = eval_and_save(last_defense_model, filtered_poisoned_testset, clean_testset, device, 
                                last_save_path)
        print(f"last_defense_model|ASR:{best_asr},ACC:{best_acc},权重保存在:{save_path}")
    else:
        best_asr, best_acc = eval_asr_acc(best_defense_model,filtered_poisoned_testset,clean_testset,device)
        last_asr, last_acc = eval_asr_acc(last_defense_model,filtered_poisoned_testset,clean_testset,device)
        print(f"best_defense_model|ASR:{best_asr},ACC:{best_acc}")
        print(f"last_defense_model|ASR:{last_asr},ACC:{last_acc}")


    end_time = time.perf_counter()
    cost_time = end_time - start_time
    hours, minutes, seconds = convert_to_hms(cost_time)
    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")

    res = {
        "PN":PN,
        "best_acc":best_acc,
        "best_asr":best_asr,
        "last_acc":last_acc,
        "last_asr":last_asr
    }

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
    attack_name ="IAD" # BadNets, IAD, Refool, WaNet, LabelConsistent
    gpu_id = 0
    r_seed = 1
    one_scene(dataset_name, model_name, attack_name, r_seed=r_seed)
    '''
    # 实验基础信息
    print("实验基础信息")
    cur_pid = os.getpid()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_name = "CleanSeedWithPoison" # CleanSeedWithPoison | Ours_SemiTrain |OursTrain
    exp_time = get_formattedDateTime()
    print("PID:",cur_pid)
    print("exp_root_dir:",exp_root_dir)
    print("exp_name:",exp_name)
    print("exp_time:",exp_time)

    # 实验保存信息
    print("实验保存信息")
    exp_save_dir = os.path.join(exp_root_dir,exp_name)
    os.makedirs(exp_save_dir,exist_ok=True)
    exp_save_file_name = f"results_{exp_time}.json"
    exp_save_path = os.path.join(exp_save_dir,exp_save_file_name)
    save_model = False
    save_json = True
    print("exp_save_path:",exp_save_path)
    print("save_model:",save_model)
    print("save_json:",save_json)

    # 实验场景信息
    print("实验场景信息")
    dataset_name_list = ["CIFAR10"] # ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18"] # ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))
    print("dataset_name_list:",dataset_name_list)
    print("model_name_list:",model_name_list)
    print("attack_name_list:",attack_name_list)
    print("r_seed_list:",r_seed_list)
    
    # 实验设备信息
    print("实验设备信息")
    gpu_id = 0
    print("gpu_id:",gpu_id)

    # 实验超参数
    print("实验超参数")
    freeze_model_flag = True # trans_seed = True
    seed_finetune_init_lr = 1e-3 # 1e-3
    seed_finetune_epochs = 100  #trans_seed = True:100; other=30
    resume_ranker_model=False
    choice_rate = 0.6
    beta = 1.0
    sigmoid_fag = False
    strict_clean= False
    seed_poisoned_num = 0
    test_seed = False
    trans_seed = True
    semi=False # 是否采用半监督训练
    print("freeze_model_flag:",freeze_model_flag)
    print("seed_finetune_init_lr:",seed_finetune_init_lr)
    print("seed_finetune_epochs:",seed_finetune_epochs)
    print("resume_ranker_model:",resume_ranker_model)
    print("choice_rate:",choice_rate)
    print("beta:",beta)
    print("sigmoid_fag:",sigmoid_fag)
    print("strict_clean:",strict_clean)
    print("seed_poisoned_num:",seed_poisoned_num)
    print("test_seed:",test_seed)
    print("trans_seed:",trans_seed)
    print("semi:",semi)

    all_start_time = time.perf_counter()
    for r_seed in r_seed_list:
        one_repeat_start_time = time.perf_counter()
        set_random_seed(r_seed)
        for dataset_name in dataset_name_list:
            for model_name in model_name_list:
                for attack_name in attack_name_list:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    print(f"\n{dataset_name}|{model_name}|{attack_name}|r_seed={r_seed}")
                    res = one_scene(dataset_name, model_name, attack_name,r_seed,save_dir=None)
                    save_experiment_result(exp_save_path, 
                           dataset_name, model_name, attack_name,r_seed,
                           res)
        one_repeat_end_time = time.perf_counter()
        one_repeat_cost_time = one_repeat_end_time - one_repeat_start_time
        hours, minutes, seconds = convert_to_hms(one_repeat_cost_time)
        print(f"\n一轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
    all_end_time = time.perf_counter()
    all_cost_time = all_end_time - all_start_time
    hours, minutes, seconds = convert_to_hms(all_cost_time)
    print(f"\n{len(r_seed_list)}轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")