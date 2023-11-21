import sys
sys.path.append("./")
import copy
import math
import time
import random
import numpy as np
import torch.nn as nn
import torch
import os
from codes import utils
from torch.utils.data import DataLoader,Dataset

# from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_BadNets_dict_state
# from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Blended_dict_state
# from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import PoisonedTrainDataset, PurePoisonedTrainDataset, PureCleanTrainDataset, PoisonedTestSet, TargetClassCleanTrainDataset,  get_dict_state as get_IAD_dict_state
# from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_LabelConsist_dict_state
# from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Refool_dict_state
# from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_WaNet_dict_state

attack_method = "WaNet" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet

if attack_method == "BadNets":
    from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "Blended":
    from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "IAD":
    from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import PoisonedTrainDataset, PurePoisonedTrainDataset, PureCleanTrainDataset, PoisonedTestSet, TargetClassCleanTrainDataset,  get_dict_state
elif attack_method == "LabelConsistent":
    from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "Refool":
    from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
elif attack_method == "WaNet":
    from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

origin_dict_state = get_dict_state()
# 本脚本全局变量
# 待变异的后门模型
backdoor_model = origin_dict_state["backdoor_model"]
clean_testset = origin_dict_state["clean_testset"]
poisoned_testset = origin_dict_state["poisoned_testset"]
pureCleanTrainDataset = origin_dict_state["pureCleanTrainDataset"]
purePoisonedTrainDataset = origin_dict_state["purePoisonedTrainDataset"]
# mutated model 保存目录
mutation_ratio = 0.01
mutation_num = 50

work_dir = f"experiments/CIFAR10/resnet18_nopretrain_32_32_3/mutates/neuron_effect_block/ratio_{mutation_ratio}_num_{mutation_num}/{attack_method}"
# 保存变异模型权重
save_dir = work_dir
utils.create_dir(save_dir)
device = torch.device('cuda:1')

def _seed_worker():
    worker_seed =  666 # torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def mutate(model, mutate_ratio):
    for count in range(mutation_num):
        model_copy = copy.deepcopy(model)
        layers = [module for module in model_copy.modules()]
        # 遍历各层
        with torch.no_grad():
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    weight = layer.weight # weight shape:our_channels, in_channels, kernel_size_0,kernel_size_1
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    mutate_num = math.ceil(in_channels*mutate_ratio)
                    selected_in_channels_id = random.sample(list(range(in_channels)),mutate_num)
                    for neuron_id in selected_in_channels_id:
                        weight[:,neuron_id,:,:] = 0
                        weight[:,neuron_id,:,:].requires_grad_()
                # if isinstance(layer, nn.Linear):
                #     weight = layer.weight # weight shape:output, input
                #     out_features, in_features = weight.shape
                #     cur_layer_neuron_num = out_features
                #     last_layer_neuron_num = in_features
                #     mutate_num = math.ceil(cur_layer_neuron_num*mutate_ratio)
                #     selected_cur_layer_neuron_ids = random.sample(list(range(cur_layer_neuron_num)),mutate_num)
                #     for neuron_id in selected_cur_layer_neuron_ids:
                #         row = weight[neuron_id]
                #         idx = torch.randperm(row.nelement())
                #         row = row.view(-1)[idx].view(row.size())
                #         row.requires_grad_()
                #         weight[neuron_id] = row
        file_name = f"model_mutated_{count+1}.pth"
        save_path = os.path.join(save_dir, file_name)
        torch.save(model_copy.state_dict(), save_path)
        print(f"变异模型:{file_name}保存成功, 保存位置:{save_path}")
    print("mutate() success")

def eval(m_i, testset):
    # 得到模型结构
    model = backdoor_model
    # 加载backdoor weights
    state_dict = torch.load(os.path.join(work_dir, f"model_mutated_{m_i}.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    total_num = len(testset)
    batch_size =128
    testset_loader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle=False,
        # num_workers=self.current_schedule['num_workers'],
        drop_last=False,
        pin_memory=False,
        worker_init_fn=_seed_worker
    )
    # 评估开始时间
    start = time.time()
    model.to(device)
    model.eval()  # put network in train mode for Dropout and Batch Normalization
    acc = torch.tensor(0., device=device) # 攻击成功率
    correct_num = 0 # 攻击成功数量
    with torch.no_grad():
        for X, Y in testset_loader:
            X = X.to(device)
            Y = Y.to(device)
            preds = model(X)
            correct_num += (torch.argmax(preds, dim=1) == Y).sum()
    acc = correct_num/total_num
    acc = round(acc.item(),3)
    end = time.time()
    print("acc:",acc)
    print(f'Total eval time: {end-start:.1f} seconds')
    print("eval() finished")
    return acc

if __name__ == "__main__":
    mutate(backdoor_model, mutation_ratio)
    asr_list = []
    acc_list = []
    for m_i in range(mutation_num):
        asr = eval(m_i+1, purePoisonedTrainDataset)
        acc = eval(m_i+1, pureCleanTrainDataset)
        asr_list.append(asr)
        acc_list.append(acc)
    print(asr_list)
    print(f"asr mean:{np.mean(asr_list)}")
    print(acc_list)
    print(f"acc mean:{np.mean(acc_list)}")
    pass
