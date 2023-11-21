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



attack_method = "LabelConsistent" # BadNets, Blended, IAD, LabelConsistent, Refool, WaNet
device = torch.device('cuda:2')

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
mutate_ratio = 0.05
mutation_num = 50
work_dir = f"/data/mml/backdoor_detect/experiments/CIFAR10/resnet18_nopretrain_32_32_3/mutates/neuron_switch/ratio_{mutate_ratio}_num_{mutation_num}/{attack_method}"
# 保存变异模型权重
save_dir = work_dir
utils.create_dir(save_dir)


def _seed_worker():
    worker_seed =  666 # torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def switch_conv2d_weights(weight, neuron_ids):
    '''
    args:
        weight:层权重 o_c, in_c, h, w =  weight.shape
        neuron_ids:切换的神经元ids
    '''
    # 拷贝一份weight出来
    weight_copy = copy.deepcopy(weight)
    # 把这些神经元id打乱
    shuffled_neuron_ids = np.random.permutation(neuron_ids)
    for neuron_id, shuffled_neuron_id in zip(neuron_ids,shuffled_neuron_ids):
        weight[:, neuron_id, :, :] = weight_copy[:,shuffled_neuron_id,:,:]
    return weight

def switch_linear_weights(weight, neuron_ids):
    weight_copy = copy.deepcopy(weight)
    shuffled_neuron_ids = np.random.permutation(neuron_ids)
    for neuron_id, shuffled_neuron_id in zip(neuron_ids,shuffled_neuron_ids):
        weight[:, neuron_id] = weight_copy[:,shuffled_neuron_id]
    return weight

def mutate(model, mutate_ratio):
    '''
    args:
        model:变异的基座模型
        mutate_ratio:变异率
    return:
        mutated model
    '''
    for count in range(mutation_num):
        # 产生一个copy model
        model_copy = copy.deepcopy(model)
        # 获得所有层
        layers = [module for module in model_copy.modules()]
        with torch.no_grad():
            # 无梯度模式下，遍历各层
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    # 如果该层为Conv2d
                    # 获得卷积层权重weight
                    weight = layer.weight # shape:(our_channels, in_channels, kernel_size_0, kernel_size_1)
                    in_channels = layer.in_channels
                    # Conv2d输入通道个数作为上层的神经元个数
                    last_layer_neuron_num = in_channels
                    # 算出上层多少个神经元需要被变异
                    mutate_num = math.ceil(last_layer_neuron_num*mutate_ratio)
                    # 获得上层神经元的索引list
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    # 从上层神经元中随机采样变异数量了的神经元id list
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids,mutate_num)
                    # 对该层权重进行变异
                    switch_conv2d_weights(weight, selected_last_layer_neuron_ids)
                if isinstance(layer, nn.Linear):
                    weight = layer.weight # weight shape:output, input
                    out_features, in_features = weight.shape
                    last_layer_neuron_num = in_features
                    mutate_num = math.ceil(last_layer_neuron_num*mutate_ratio)
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids, mutate_num)
                    switch_linear_weights(weight, selected_last_layer_neuron_ids)
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


def review_model():
    model_review = utils.ModelReview()
    model_review.set_model(backdoor_model)
    model_review.see_layers()


if __name__ == "__main__":
    # review_model()
    
    mutate(backdoor_model, mutate_ratio)
    acc_list = []
    asr_list = []
    for m_i in range(mutation_num):
        acc = eval(m_i+1, pureCleanTrainDataset)
        asr = eval(m_i+1, purePoisonedTrainDataset)
        acc_list.append(acc)
        asr_list.append(asr)
    print(acc_list,"\n")
    print(f"ACC mean:{np.mean(acc_list)}","\n")
    print(asr_list,"\n")
    print(f"ASR mean:{np.mean(asr_list)}", "\n")
    pass
