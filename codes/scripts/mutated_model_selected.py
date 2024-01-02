import queue

import torch
from cliffs_delta import cliffs_delta

from codes.eval_model import EvalModel
from codes.utils import priorityQueue_2_list


'''
下边两个选择法都是寻找能够在各自指标上区分suspected_dataset和non_suspected_dataset的变异模型
'''

def select_by_suspected_nonsuspected_acc_dif(model_struct, weight_file_path_list:list, suspected_dataset, non_suspected_dataset, device):
    '''
    mutated model在怀疑集上的acc - 非怀疑集上acc作为dif.
    dif越大,该变异模型被保留的优先级越高。
    因为,说明该变异模型在acc度量上,在怀疑集和非怀疑集上具备较大差异。且在怀疑集上有更大的acc,说明该
    变异模型在怀疑集上更加鲁棒。我们直观认为,是由于怀疑集上的backdoor样本更加鲁棒造成的。

    因此,对应的检测法可以为
    优先级法:
    1:标签保持率
        怀疑集 -> 被选择的变异模型集 -> 标签保持率越大的实例越可能为backdoor实例
    2:熵
        怀疑集 -> 被选择的变异模型集 -> 熵越小的实例越可能为backdoor实例
    判定法:
    1:标签保持率
        (1)非怀疑集 -> 被选择的变异模型集 -> 统计得标签保持率list -> 置信区间
        (2)怀疑集 -> 被选择的变异模型集 -> 实例标签保持率大于置信区间上界 -> 判定为backdoor实例
    2:熵
        (1)非怀疑集 -> 被选择的变异模型集 -> 熵list -> 置信区间
        (2)怀疑集 -> 被选择的变异模型集 -> 熵小于置信区间上界 -> 判定为backdoor实例
    '''
    selected_weight_file_path_list = []
    sorted_weight_file_path_list = []
    q = queue.PriorityQueue()
    for m_i, weight_file_path in enumerate(weight_file_path_list):
        model_struct.load_state_dict(torch.load(weight_file_path, map_location="cpu"))
        e = EvalModel(model_struct, suspected_dataset, device)
        acc_suspected = e._eval_acc()
        e = EvalModel(model_struct, non_suspected_dataset, device)
        acc_non_suspected = e._eval_acc()
        priority = acc_suspected - acc_non_suspected # 越大优先级越高
        q.put((-priority, weight_file_path))
    q_list = priorityQueue_2_list(q)
    for item in q_list:
        sorted_weight_file_path_list.append(item[1])
    selected_weight_file_path_list = sorted_weight_file_path_list[:50]
    return selected_weight_file_path_list

def select_by_suspected_nonsuspected_confidence_distribution_dif(model_struct, weight_file_path_list:list, suspected_dataset, non_suspected_dataset, device):
    '''
    统计mutated model在怀疑集上置信度分布与非怀疑集上置信度分布。
    两者分布具有统计学差异,如cliffs_delta的绝对值越大,说明两者分布差异越大。
    将分布差异越大该变异模型被保留的优先级越高。
    因为,差异越大说明该变异模型的置信度输出在两个集上具有区分度。该模型的confidence可以作为样本的一个特征。

    因此,对应的检测法可以为
    优先级法:
    1: 非怀疑集confidence中心点距离
        (1)非怀疑集 -> 被选择的变异模型集 -> 样本集的平均confidence -> 作为中心point
        (2)怀疑集 -> 被选择的变异模型集 -> 样本在变异模型上的confidence到中心point距离越大越有可能为backdoor实例
    判定法:
    2: 非怀疑集到confidence中心点距离置信区间
        (1)非怀疑集 -> 被选择的变异模型集 -> 样本集的平均confidence -> 作为中心point,同时计算置信区间
        (2)怀疑集 -> 被选择的变异模型集 -> 样本在变异模型上的confidence到中心point距离大于置信区间上界 => 判定为backdoor实例
    '''
    selected_weight_file_path_list = []
    q = queue.PriorityQueue()
    for m_i, weight_file_path in enumerate(weight_file_path_list):
        model_struct.load_state_dict(torch.load(weight_file_path, map_location="cpu"))
        e = EvalModel(model_struct, suspected_dataset, device)
        suspected_confidence_list = e._get_confidence_list()
        sorted_suspected_confidence_list = sorted(suspected_confidence_list)
        e = EvalModel(model_struct, non_suspected_dataset, device)
        non_suspected_confidence_list = e._get_confidence_list()
        sorted_non_suspected_confidence_list = sorted(non_suspected_confidence_list)
        d,info = cliffs_delta(sorted_suspected_confidence_list, sorted_non_suspected_confidence_list)
        priority = abs(d) # 越大优先级越高
        q.put((-priority, weight_file_path))
    sorted_weight_file_path_list = priorityQueue_2_list(q)
    selected_weight_file_path_list = sorted_weight_file_path_list[:50]
    return selected_weight_file_path_list


