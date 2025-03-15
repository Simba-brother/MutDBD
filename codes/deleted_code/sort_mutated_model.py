import queue
import random
from codes import config
from codes.scripts.dataset_constructor import *
from codes.tools import EvalModel
from codes.utils import priorityQueue_2_list


def sort_mutated_models(
        model_struct, 
        mutation_weights_path_list,
        target_class_clean_set, 
        target_class_poisoned_set,
        device
    ):
    '''
    对变异模型进行排序
    Args:
        model_struct: 变异模型结构
        target_class_clean_set: 目标类别中的干净集，用于挑选种子
        target_class_poisoned_set:
    Return:
        sorted_mutated_weights_list: 排好序的变异权重list
    '''
    # 获得种子
    # 优先级队列q,值越小优先级越高
    q = queue.PriorityQueue()
    # 每个类别选10个clean samples，作为我们的种子个数
    seed_num = 10*config.class_num
    # clean set ids
    ids = list(range(len(target_class_clean_set)))
    # 打乱编号顺序
    random.shuffle(ids)
    # 选择seed_ids
    selected_ids = ids[0:seed_num]
    # 获得remain_ids
    remain_ids = list(set(ids) - set(selected_ids))
    clean_seed_dataset = ExtractDatasetByIds(target_class_clean_set, selected_ids)
    clean_remain_dataset = ExtractDatasetByIds(target_class_clean_set, remain_ids)
    remain_dataset = CombinDataset(clean_remain_dataset,target_class_poisoned_set)

    for mutation_weights_path in  mutation_weights_path_list:
        weights = torch.load(mutation_weights_path, map_location="cpu")
        model_struct.load_state_dict(weights)
        e = EvalModel(model_struct, clean_seed_dataset, device)
        acc_seed = e._eval_acc()
        e = EvalModel(model_struct, remain_dataset, device)
        acc_remain = e._eval_acc()
        priority = acc_seed - acc_remain # 越小,优先级越高,因为越小有可能acc_seed很低，说明该变异模型在clean seed上预测的很乱。同时acc_remain很高，说明该变异模型在poisoned samples上预测的很稳定
        item = (priority, mutation_weights_path)
        q.put(item)
            
    priority_list = priorityQueue_2_list(q)
    sorted_mutated_weights_path_list = [weights_path for priority, weights_path in priority_list]
    return sorted_mutated_weights_path_list

