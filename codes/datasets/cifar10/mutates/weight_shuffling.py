import sys
sys.path.append("./")
import copy
import math
import random
import numpy as np
import torch.nn as nn
import torch

# from codes.datasets.cifar10.attacks.badnets_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_BadNets_dict_state
# from codes.datasets.cifar10.attacks.Blended_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Blended_dict_state
# from codes.datasets.cifar10.attacks.IAD_resnet18_nopretrain_32_32_3 import PoisonedTrainDataset, PurePoisonedTrainDataset, PureCleanTrainDataset, PoisonedTestSet, TargetClassCleanTrainDataset,  get_dict_state as get_IAD_dict_state
# from codes.datasets.cifar10.attacks.LabelConsistent_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_LabelConsist_dict_state
from codes.datasets.cifar10.attacks.Refool_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_Refool_dict_state
# from codes.datasets.cifar10.attacks.WaNet_resnet18_nopretrain_32_32_3 import PureCleanTrainDataset, PurePoisonedTrainDataset, get_dict_state as get_WaNet_dict_state

dict_state = get_Refool_dict_state()
backdoor_model = dict_state["backdoor_model"]
mutate_ratio = 0.01
random.seed(666)
np.random.seed(666)

def mutate(model, mutate_ratio):
    model_copy = copy.deepcopy(model)
    layers = [module for module in model_copy.modules()]
    # 遍历各层
    with torch.no_grad():
        for layer in layers[1:]:
            if isinstance(layer, nn.Conv2d):
                weight = layer.weight # weight shape:our_channels, in_channels, kernel_size_0,kernel_size_1
                in_channels = layer.in_channels
                out_channels = layer.out_channels
                cur_layer_neuron_num = out_channels
                last_layer_neuron_num = in_channels
                mutate_num = math.ceil(cur_layer_neuron_num*mutate_ratio)
                selected_cur_layer_neuron_ids = random.sample(list(range(cur_layer_neuron_num)),mutate_num)
                selected_cur_layer_neuron_ids_shuffled = list(np.random.permutation(selected_cur_layer_neuron_ids))
                temp_w_list = []
                for neuron_id in selected_cur_layer_neuron_ids_shuffled:
                    temp_w_list.append(weight[neuron_id])
                    
                for neuron_id, temp_w in zip(selected_cur_layer_neuron_ids,temp_w_list):
                    weight[neuron_id] = temp_w
            if isinstance(layer, nn.Linear):
                weight = layer.weight # weight shape:our_channels, in_channels, kernel_size_0,kernel_size_1
                print("fla")
    return False

if __name__ == "__main__":
    mutate(backdoor_model, mutate_ratio)














if __name__=="__main__":
    pass