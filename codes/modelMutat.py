import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
global_seed = 666
torch.manual_seed(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)

def add_Gaussian_perturbation(weight, neuron_ids, scale):
    neuron_num = len(neuron_ids)
    out_features = weight.shape[0]
    normal_size = neuron_num*out_features
    disturb_array = np.random.normal(scale=scale, size=normal_size) 
    start_idx = 0
    for neuron_id in neuron_ids:
        col = weight[:,neuron_id]
        end_idx = start_idx + out_features
        cur_disturb_array = disturb_array[start_idx:end_idx]
        start_idx = end_idx
        col += cur_disturb_array
        weight[:,neuron_id] = col
    return weight


def shuffle_conv2d_weights(weight, neuron_id):
    o_c, in_c, h, w =  weight.shape
    weight = weight.reshape(o_c, in_c * h * w)
    row = weight[neuron_id]
    idx = torch.randperm(row.nelement())
    row = row.view(-1)[idx].view(row.size())
    row.requires_grad_()
    weight[neuron_id] = row
    weight = weight.reshape(o_c, in_c, h, w)
    return weight

def switch_linear_weights(weight, neuron_ids):
    weight_copy = copy.deepcopy(weight)
    shuffled_neuron_ids = np.random.permutation(neuron_ids)
    for neuron_id, shuffled_neuron_id in zip(neuron_ids,shuffled_neuron_ids):
        weight[:, neuron_id] = weight_copy[:,shuffled_neuron_id]
    return weight

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
        weight[neuron_id,:, :, :] = weight_copy[shuffled_neuron_id,:,:,:]
    return weight

class ModelMutat(object):
    def __init__(self, original_model, mutation_ratio):
        self.original_model = original_model
        self.mutation_ratio = mutation_ratio
        
    def _gf_mut(self, scale=5):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        with torch.no_grad():
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    weight = layer.weight
                    out_features, in_features = weight.shape
                    last_layer_neuron_num = in_features
                    selected_neuron_num = math.ceil(last_layer_neuron_num*self.mutation_ratio)
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids,selected_neuron_num)
                    add_Gaussian_perturbation(weight, selected_last_layer_neuron_ids, scale)
        return model_copy
    
    def _neuron_activation_inverse(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        # 遍历各层
        with torch.no_grad():
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    weight = layer.weight # weight shape:our_channels, in_channels, kernel_size_0,kernel_size_1
                    neuron_num = layer.out_channels
                    selected_neuron_num = math.ceil(neuron_num*self.mutation_ratio)
                    neuron_ids = list(range(neuron_num))
                    selected_neuron_ids = random.sample(neuron_ids,selected_neuron_num)
                    for neuron_id in selected_neuron_ids:
                        weight[neuron_id,:,:,:] *= -1
                if isinstance(layer, nn.Linear):
                    weight = layer.weight # weight shape:output, input
                    out_features, in_features = weight.shape
                    cur_layer_neuron_num = out_features
                    last_layer_neuron_num = in_features
                    selected_neuron_num = math.ceil(last_layer_neuron_num*self.mutation_ratio)
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids,selected_neuron_num)
                    for neuron_id in selected_last_layer_neuron_ids:
                        weight[:,neuron_id] *= -1
        return model_copy
    
    def _neuron_block(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        # 遍历各层
        with torch.no_grad():
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    weight = layer.weight # weight shape:out_channels, in_channels, kernel_size_0,kernel_size_1
                    neuron_num = layer.out_channels
                    selected_neuron_num = math.ceil(neuron_num*self.mutation_ratio)
                    selected_neuron_ids = random.sample(list(range(neuron_num)),selected_neuron_num)
                    for neuron_id in selected_neuron_ids:
                        weight[neuron_id,:,:,:] = 0
                        weight[neuron_id,:,:,:].requires_grad_()

                if isinstance(layer, nn.Linear):
                    weight = layer.weight # weight shape:output, input
                    out_features, in_features = weight.shape
                    last_layer_neuron_num = in_features
                    selected_neuron_num = math.ceil(last_layer_neuron_num*self.mutation_ratio)
                    selected_last_layer_neuron_ids = random.sample(list(range(last_layer_neuron_num)),selected_neuron_num)
                    for neuron_id in selected_last_layer_neuron_ids:
                        weight[:,neuron_id] = 0
        return model_copy
    
    def _neuron_switch(self):
        # 产生一个copy model
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        # 获得所有层
        layers = [module for module in model_copy.modules()]
        with torch.no_grad():
            # 无梯度模式下，遍历各层
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    # 如果该层为Conv2d
                    # 获得卷积层权重weight
                    weight = layer.weight # shape:(out_channels, in_channels, kernel_size_0, kernel_size_1)
                    neuron_num = layer.out_channels
                    # Conv2d输入通道个数作为上层的神经元个数
                    # 算出上层多少个神经元需要被变异
                    selected_neuron_num = math.ceil(neuron_num*self.mutation_ratio)
                    # 获得上层神经元的索引list
                    neuron_ids = list(range(neuron_num))
                    # 从上层神经元中随机采样变异数量了的神经元id list
                    selected_neuron_ids = random.sample(neuron_ids,selected_neuron_num)
                    # 对该层权重进行变异
                    switch_conv2d_weights(weight, selected_neuron_ids)
                if isinstance(layer, nn.Linear):
                    weight = layer.weight # weight shape:output, input
                    # bias = layer.bias
                    out_features, in_features = weight.shape
                    last_layer_neuron_num = in_features
                    selected_neuron_num = math.ceil(last_layer_neuron_num*self.mutation_ratio)
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids, selected_neuron_num)
                    switch_linear_weights(weight, selected_last_layer_neuron_ids)
        return model_copy
    
    def _weight_shuffling(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        # 遍历各层
        with torch.no_grad():
            for layer in layers[1:]:
                if isinstance(layer, nn.Conv2d):
                    weight = layer.weight # weight shape:our_channels, in_channels, kernel_size_0,kernel_size_1
                    neuron_num = layer.out_channels
                    selected_neuron_num = math.ceil(neuron_num*self.mutation_ratio)
                    selected_neuron_ids = list(range(selected_neuron_num))
                    selected_cur_layer_neuron_ids = random.sample(selected_neuron_ids,selected_neuron_num)
                    for neuron_id in selected_cur_layer_neuron_ids:
                        shuffle_conv2d_weights(weight, neuron_id)
                if isinstance(layer, nn.Linear):
                    weight = layer.weight # weight shape:output, input
                    out_features, in_features = weight.shape
                    # cur_layer_neuron_num = out_features
                    last_layer_neuron_num = in_features
                    selected_neuron_num = math.ceil(last_layer_neuron_num*self.mutation_ratio)
                    last_layer_neuron_ids = list(range(last_layer_neuron_num))
                    selected_last_layer_neuron_ids = random.sample(last_layer_neuron_ids,selected_neuron_num)
                    for neuron_id in selected_last_layer_neuron_ids:
                        col = weight[:,neuron_id]
                        idx = torch.randperm(col.nelement())
                        col = col.view(-1)[idx].view(col.size())
                        col.requires_grad_()
                        weight[:,neuron_id] = col
        return model_copy
    


class ModelMutat_2(object):
    def __init__(self, original_model, mutation_rate):
        self.original_model = original_model
        self.mutation_rate = mutation_rate
        
    def _gf_mut(self, scale=None):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        linear_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        mutated_layer = linear_layers[-2]
        with torch.no_grad():
            layer = mutated_layer
            weight = layer.weight
            if scale is None:
                scale = np.std(weight.tolist())
            out_features, in_features = weight.shape
            out_neuron_num = out_features
            selected_neuron_num = math.ceil(out_neuron_num*self.mutation_rate)
            out_neuron_id_list = list(range(out_neuron_num))
            random.shuffle(out_neuron_id_list)
            selected_out_neuron_id_list = out_neuron_id_list[:selected_neuron_num]
            # add GF
            normal_size = selected_neuron_num*in_features
            disturb_array = np.random.normal(scale=scale, size=normal_size) 
            start_idx = 0
            for selected_out_neuron_id in selected_out_neuron_id_list:
                row = weight[selected_out_neuron_id:]
                end_idx = start_idx + in_features
                cur_disturb_array = disturb_array[start_idx:end_idx]
                start_idx = end_idx
                row += cur_disturb_array
                weight[selected_out_neuron_id,:] = row
        return model_copy
    
    def _neuron_activation_inverse(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        linear_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        mutated_layer = linear_layers[-2]
        # 遍历各层
        with torch.no_grad():
            layer = mutated_layer
            weight = layer.weight # weight shape:output, input
            out_features, in_features = weight.shape
            out_neuron_num = out_features
            selected_neuron_num = math.ceil(out_neuron_num*self.mutation_rate)
            out_neuron_id_list = list(range(selected_neuron_num))
            random.shuffle(out_neuron_id_list)
            selected_out_neuron_id_list = out_neuron_id_list[:selected_neuron_num]
            for selected_out_neuron_id in selected_out_neuron_id_list:
                weight[selected_out_neuron_id,:] *= -1
        return model_copy
    
    def _neuron_block(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        linear_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        mutated_layer = linear_layers[-2]
        # 遍历各层
        with torch.no_grad():
            layer = mutated_layer
            weight = layer.weight # weight shape:output, input
            out_features, in_features = weight.shape
            out_neuron_num = out_features
            selected_neuron_num = math.ceil(out_neuron_num*self.mutation_rate)
            out_neuron_id_list = list(range(selected_neuron_num))
            random.shuffle(out_neuron_id_list)
            selected_out_neuron_id_list = out_neuron_id_list[:selected_neuron_num]
            for selected_out_neuron_id in selected_out_neuron_id_list:
                weight[selected_out_neuron_id,:] *= 0
        return model_copy
    
    def _neuron_switch(self):
        # 产生一个copy model
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        # 获得所有层
        layers = [module for module in model_copy.modules()]
        linear_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        mutated_layer = linear_layers[-2]
        with torch.no_grad():
            layer = mutated_layer
            weight = layer.weight # weight shape:output, input
            out_features, in_features = weight.shape
            # bias = layer.bias
            out_neuron_num = out_features
            selected_neuron_num = math.ceil(out_neuron_num*self.mutation_rate)
            out_neuron_id_list = list(range(selected_neuron_num))
            random.shuffle(out_neuron_id_list)
            selected_out_neuron_id_list = out_neuron_id_list[:selected_neuron_num]
            # switch
            shuffled_selected_out_neuron_id_list = np.random.permutation(selected_out_neuron_id_list)
            weight_copy = copy.deepcopy(weight)
            for neuron_id, shuffled_neuron_id in zip(selected_out_neuron_id_list,shuffled_selected_out_neuron_id_list):
                weight[neuron_id,:] = weight_copy[shuffled_neuron_id,neuron_id]
        return model_copy
    
    def _weight_shuffling(self):
        model_copy = copy.deepcopy(self.original_model)
        model_copy.to(torch.device("cpu"))
        layers = [module for module in model_copy.modules()]
        linear_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        mutated_layer = linear_layers[-2]
        # 遍历各层
        with torch.no_grad():
            layer = mutated_layer
            weight = layer.weight # weight shape:output, input
            out_features, in_features = weight.shape
            out_neuron_num = out_features
            selected_neuron_num = math.ceil(out_neuron_num*self.mutation_rate)
            out_neuron_id_list = list(range(selected_neuron_num))
            random.shuffle(out_neuron_id_list)
            selected_out_neuron_id_list = out_neuron_id_list[:selected_neuron_num]
            for neuron_id in selected_out_neuron_id_list:
                row = weight[neuron_id,:]
                idx = torch.randperm(row.nelement())
                row = row.view(-1)[idx].view(row.size())
                row.requires_grad_()
                weight[neuron_id,:] = row
        return model_copy
    

# dataset_name = "cifar10"    
# model_name = "resnet18"
# attack_name = "badnets"
# if dataset_name == "cifar10":
#     if model_name == "resnet18":
#         if attack_name == "badnets":
#             from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state 
# origin_dict_state = get_dict_state()
# backdoor_model = origin_dict_state["backdoor_model"]
# clean_testset = origin_dict_state["clean_testset"]
# poisoned_testset = origin_dict_state["poisoned_testset"]
# pureCleanTrainDataset = origin_dict_state["pureCleanTrainDataset"]
# purePoisonedTrainDataset = origin_dict_state["purePoisonedTrainDataset"]
# poisoned_trainset = origin_dict_state["poisoned_trainset"]

# def start_mutate():
        
#     mutation_model_num = 50
#     mutation_ratio_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#     for mutation_ratio in mutation_ratio_list:
#         for m_i in range(mutation_model_num):
#             modelMutat = ModelMutat(backdoor_model, mutation_ratio, poisoned_trainset)
#             mutated_model = modelMutat._gf_mut(scale=5)
#             evalModel = EvalModel(mutated_model, poisoned_trainset)
#             acc = evalModel._eval()