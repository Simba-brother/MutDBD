# coding:utf8
'''
This module only supports neuron and weight mutation.
Ma, Lei, et al. "DeepMutation: Mutation Testing of Deep Learning Systems."
code:https://github.com/Simba-brother/m_testing_adversatial_sample
'''

import torch
import copy
import numpy as np
import math


class MutaionOperator(object):
    def __init__(self, ration, model, verbose=True,device=torch.device("cpu")):
        '''

        :param ration:
        :param model:
        :param acc_tolerant: 种子模型的90%
        :param verbose: print the mutated detail or not. like the number of weights to be mutated with layer
        :param test:
        :param device: torch.device("cuda:0")|torch.device("cpu")
        '''
        # 变异比例
        self.ration = ration
        # 原始模型
        self.original_model = model.to(device)
        self.device = device
        self.verbose = verbose

    def gf(self, std=None):
        '''
        Gaussian Fuzzing is a model mutation method in weight level
        :param std: the scale parameter of Gaussian Distribution
        :return: a mutated model
        '''
        # 深度拷贝出一个模型
        mutation_model = copy.deepcopy(self.original_model)
        # 总权重数量
        num_weights = 0
        # 总层数
        num_layers = 0  # including the bias
        # 存储每一层的标准差
        std_layers = [] # store each the standard deviation of paramets of a layer
        # 遍历每一层的参数
        for param in mutation_model.parameters():
            # 该层参数数量
            num_weights += (param.data.view(-1)).size()[0] # 模型总权重数量 view()函数中也可以传入-1作为参数。Tensor.view(-1)会让原张量直接展开成一维结构
            # 层数+1
            num_layers += 1
            # 该层std
            std_layers.append(param.data.std().item())
        # 从总权重list中选择要变异的权重
        indices = np.random.choice(num_weights, int(num_weights * self.ration), replace=False)
        weights_count = 0 # 每一层权重索引的起始索引
        for idx_layer, param in enumerate(mutation_model.parameters()):
            # 当前层权重原本的形状
            shape = param.data.size()
            # 当前层的权重数量
            num_weights_layer = (param.data.view(-1)).size()[0]
            # 当前层中被变异的权重索引
            mutated_indices = set(indices) & set(
                np.arange(weights_count, weights_count + num_weights_layer))

            if mutated_indices:
                # 如果有
                mutated_indices = np.array(list(mutated_indices))
                #########################
                # project the global index to the index of current layer
                #########################
                # 当前层的待变异的权重的索引
                mutated_indices = mutated_indices - weights_count
                # 当前层的权重（to numpy）
                current_weights = param.data.cpu().view(-1).numpy() 
                #####################
                #  Note: there is a slight difference from the original paper,in which a new
                #  value is generated via Gaussian distribution with the original single weight as the expectation,
                #  while we use the mean of all potential mutated weights as the expectation considering the time-consuming.
                #  In a nut shell, we yield all the mutated weights at once instead of one by one
                #########################
                # 当前层的权重均值
                avg_weights = np.mean(current_weights)
                # 当前层的标准差
                current_std = std if std else std_layers[idx_layer]
                # 获得当前层变异权重的值
                mutated_weights = np.random.normal(avg_weights, current_std, mutated_indices.size)

                current_weights[mutated_indices] = mutated_weights 
                new_weights = torch.Tensor(current_weights).reshape(shape)
                param.data = new_weights.to(self.device)
            if self.verbose:
                print(">>:mutated weights in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_indices),
                                                                         num_weights_layer))
            # 当前层权重起始索引更新
            weights_count += num_weights_layer

        return mutation_model

    def ws(self):
        '''
        打乱的神经元与前一层连接的权重
        Weight Shuffling. Shuffle selected weights
        Randomly select neurons and shuffle the weights of each neuron.即,打乱选择的神经元的权重
        The key point is to select the neurons and record the weights of its connection with previous layer
        For a regular layer,say full connected layer, it is a easy task, but it may be not straight to select the
        neurons in convolutional layer. we could make follow assumptions:
        1. The number of neurons in convolutional layer is equal to the number of its output elements
        2. Given the parameter sharing in conv layer,  the neuron of each
            slice in output volume has the same weights(i.e, the corresponding slice of the conv kernel)
        Hence, it is impossible to shuffle the weights of a neuron without changing others' weights which are in the same
        slice.
        To this end, instead of neurons, we shuffle the weights of certain slices.
        Note: we don't take the bias into account.
        :return: a mutated model
        '''
        unique_neurons = 0 # 所有神经元个数
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for param in mutation_model.parameters():
            shape = param.size()
            dim = len(shape)
            if dim > 1: # 排除bias层
                unique_neurons += shape[0]
        # 从整数范围内随机选择，不放回
        indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
        neurons_count = 0
        # 遍历层
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            # 获得当前层的维度
            dim = len(shape)
            # skip the bias
            if dim > 1:
                # 当前层的神经元个数
                unique_neurons_layer = shape[0]
                mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                if mutated_neurons:
                    # 如果当前层存在变异的神经元数量，映射局部索引
                    mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                    for neuron in mutated_neurons:
                        # 记录当前待变异神经元权重原始形状
                        ori_shape = param.data[neuron].size()
                        old_data = param.data[neuron].view(-1).cpu().numpy()
                        # shuffle
                        shuffle_idx = np.arange(len(old_data))
                        # 把这个神经元的权重打乱
                        np.random.shuffle(shuffle_idx)
                        new_data = old_data[shuffle_idx]
                        new_data = torch.Tensor(new_data).reshape(ori_shape)
                        param.data[neuron] = new_data.to(self.device)
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
                neurons_count += unique_neurons_layer

        return mutation_model

    def ns(self, skip=10):
        '''
        Neuron Switch.
        The NS operator switches two neurons within a layer to exchange their roles and inﬂuences for next layers.即,对同一层的神经元进行切换
        Note: we don't take the bias into account and set a constraint that the number of neurons( for regular layer)
        or filters( for convolution layer) of a layer should be at least greater than a given threshold since at least two
        neurons or filters are involved in a switch. We set 10 as the default value.
        The switch process is limited within a layer.
        :param skip: the threshold of amount of neurons in layer,根据分类层数量决定
        :return:
        '''
        # 拷贝出一个变异模型
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        # 遍历模型每一层
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            dim = len(shape)
            unique_neurons_layer = shape[0]
            # skip the bias
            if dim > 1 and unique_neurons_layer >= skip: # 神经元个数小于10的不切换
                temp = unique_neurons_layer * self.ration
                num_mutated = math.floor(temp) if temp > 2. else math.ceil(temp)
                mutated_neurons = np.random.choice(unique_neurons_layer,
                                                   int(num_mutated), replace=False)
                switch = copy.copy(mutated_neurons)
                np.random.shuffle(switch)
                param.data[mutated_neurons] = param.data[switch]
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
        return mutation_model

    def nai(self):
        '''
        The NAI operator tries to invert the activation status of a neuron,
        which can be achieved by changing the sign of the output value of
        a neuron before applying its activation function.
        Note: In this operator, we take the bias into account,but we don't regard the bias unit as a neuron
        :return:
        '''
        unique_neurons = 0
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for param in mutation_model.parameters():
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons += shape[0]
        # select which neurons should be to inverted.
        indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
        neurons_count = 0
        last_mutated_neurons = []
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons_layer = shape[0]
                mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                if mutated_neurons:
                    mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                    param.data[mutated_neurons] = -1 * param.data[mutated_neurons]
                    last_mutated_neurons = mutated_neurons
                neurons_count += unique_neurons_layer
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
            else:
                # invert the bias
                param.data[last_mutated_neurons] = -1 * param.data[last_mutated_neurons]
                last_mutated_neurons = []

        return mutation_model


    def nb(self):
        '''
        neuron_block
        '''
        unique_neurons = 0
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for param in mutation_model.parameters():
            shape = param.size()
            dim = len(shape)
            if dim > 1: # 跳过bias层
                unique_neurons += shape[0]
        # select which neurons should be to inverted.选择索引
        indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
        neurons_count = 0
        last_mutated_neurons = []
        # 遍历层
        for idx_layer, param in enumerate(mutation_model.parameters()):
            # 该层参数形状
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons_layer = shape[0]
                # 该层要变异的神经元
                mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                if mutated_neurons:
                    mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                    param.data[mutated_neurons] = 0
                    last_mutated_neurons = mutated_neurons # 该层变异的神经元赋值给last_mutated_neurons变量
                neurons_count += unique_neurons_layer
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
            else:
                # 如果该层是bias
                # invert the bias
                param.data[last_mutated_neurons] = 0
                last_mutated_neurons = []

        return mutation_model
        


if __name__ == '__main__':
    pass
