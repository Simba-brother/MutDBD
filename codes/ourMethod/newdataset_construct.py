import math
from codes.scripts.dataset_constructor import CombinDataset,ExtractDatasetByIds

def get_train_dataset(priority_list, cut_off, target_class_clean_set, purePoisonedTrainDataset, no_target_class_dataset):
    '''
    得到剔除了可疑木马样本的训练集
    Args:
        priority_list:优先级list,结构[(entropy, ground_truth, id),...]
        cut_off:剔除的cutoff
        target_class_clean_set: target class中的真正clean的
        purePoisonedTrainDataset: target class中的真正poisoned的
    Return:
        new_trainset
    '''
    id_list = [item[2] for item in priority_list]
    gt_list = [item[1] for item in priority_list]

    cut_point = math.ceil(len(id_list)*cut_off)
    # 剔除掉队头
    remain_ids = id_list[cut_point:]
    remain_gts = gt_list[cut_point:] # true表示木马
    combin_dataset = CombinDataset(target_class_clean_set,purePoisonedTrainDataset)
    new_target_class_trainset = ExtractDatasetByIds(combin_dataset,remain_ids)
    new_trainset = CombinDataset(new_target_class_trainset,no_target_class_dataset)
    print(f"保留的训练集size:{len(new_trainset)},其中还有木马样本数量:{sum(remain_gts)}")
    return new_trainset

