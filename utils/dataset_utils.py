
from torch.utils.data import DataLoader,Subset
from defense.our.sample_select import sort_sample_id

def get_class_num(dataset_name):
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    else:
        raise ValueError("Invalid input")
    return class_num

def filter_poisonedSet(clean_set,poisoned_set,target_class_idx):
    # 从poisoned_testset中剔除原来就是target class的数据
    clean_testset_loader = DataLoader(
                clean_set, # 非预制
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    clean_testset_label_list = []
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(clean_set)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != target_class_idx:
            filtered_ids.append(sample_id)
    filtered_poisoned_set = Subset(poisoned_set,filtered_ids)
    return filtered_poisoned_set

def split_method(
        ranker_model,
        poisoned_trainset,
        poisoned_ids,
        device,
        class_rank = None,
        choice_rate = 0.5
        ):
    # poisoned_trainset_loader = DataLoader(
    #     poisoned_trainset,
    #     batch_size = 256,
    #     shuffle=False,
    #     num_workers=4,
    #     drop_last=False,
    #     pin_memory=True
    # )
    ranked_sample_id_list, _, _ = sort_sample_id(
                            ranker_model,
                            device,
                            poisoned_trainset,
                            poisoned_ids,
                            class_rank)
    num = int(len(ranked_sample_id_list)*choice_rate)
    choiced_sample_id_list = ranked_sample_id_list[:num]
    remain_sample_id_list = ranked_sample_id_list[num:]
    # 统计一下污染的含量
    # 干净池总数
    choiced_num = len(choiced_sample_id_list)
    # 干净池中的中毒数量
    p_count = 0
    for choiced_sample_id in choiced_sample_id_list:
        if choiced_sample_id in poisoned_ids:
            p_count += 1
    # 干净池的中毒比例
    poisoning_rate = round(p_count/choiced_num, 2)
    return p_count, choiced_num, poisoning_rate
    '''
    choicedSet = Subset(poisoned_trainset,choiced_sample_id_list)
    remainSet = Subset(poisoned_trainset,remain_sample_id_list)
    
    return choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list
    '''