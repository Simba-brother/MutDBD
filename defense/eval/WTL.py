'''
统计所有场景的W/T/L
'''
import os
import yaml
import torch
from utils.calcu_utils import compare_WTL,compare_avg
from utils.common_utils import read_yaml,get_formattedDateTime
from models.model_loader import get_model
from datasets.posisoned_dataset import get_all_dataset
from utils.model_eval_utils import eval_asr_acc,EvalModel
import joblib
from tqdm import tqdm
from mid_data_loader import get_class_rank, get_our_method_state, get_asd_method_state,get_backdoor_data
from utils.dataset_utils import split_method,get_class_num

def get_acc_asr_pn(
        defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = None, choice_rate=0.5):
    # ACC和ASR
    em = EvalModel(defence_model, clean_testset, device)
    acc = em.eval_acc()
    em = EvalModel(defence_model, filtered_poisoned_testset, device)
    asr = em.eval_acc()
    # 中毒样本切分结果
    p_num, choiced_num, poisoning_rate = split_method(
        select_model,
        poisoned_trainset,
        poisoned_ids,
        device,
        class_rank = class_rank,
        choice_rate = choice_rate
        )
    res = {"acc":acc,"asr":asr,"p_num":p_num}
    return res

def our_unit_res(dataset_name, model_name, attack_name, random_seed, 
             poisoned_trainset, poisoned_ids,
             filtered_poisoned_testset, clean_testset,
             device):
    # 过滤掉原来target class的样本
    
    # OurRes
    defensed_state_dict_path, selected_state_dict_path = get_our_method_state(dataset_name, model_name, attack_name, random_seed)
    defence_model = get_model(dataset_name,model_name)
    select_model = get_model(dataset_name,model_name)
    defence_model.load_state_dict(torch.load(defensed_state_dict_path,map_location="cpu"))
    select_model.load_state_dict(torch.load(selected_state_dict_path,map_location="cpu"))
    # seed微调后排序一下样本
    class_rank = get_class_rank(dataset_name, model_name, attack_name)

    our_res = get_acc_asr_pn(defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = class_rank, choice_rate=0.6)
    return our_res

def asd_unit_res(dataset_name, model_name, attack_name, random_seed, 
             poisoned_trainset, poisoned_ids,
             filtered_poisoned_testset, clean_testset,
             device):
    # ASDRes
    defensed_state_dict_path, selected_state_dict_path = get_asd_method_state(dataset_name, model_name, attack_name, random_seed)
    defence_model = get_model(dataset_name,model_name)
    select_model = get_model(dataset_name,model_name)
    defence_model.load_state_dict(torch.load(defensed_state_dict_path,map_location="cpu")["model_state_dict"])
    select_model.load_state_dict(torch.load(selected_state_dict_path,map_location="cpu"))

    ASD_res = get_acc_asr_pn(defence_model, select_model, device, 
        clean_testset, filtered_poisoned_testset, poisoned_trainset, poisoned_ids,
        class_rank = None, choice_rate=0.5)
    return ASD_res

def main_scene():
    # 后门信息
    backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 数据集
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    
    # 10次重复实验记录
    our_acc_list = []
    our_asr_list = []
    our_p_num_list = []

    asd_acc_list = []
    asd_asr_list = []
    asd_p_num_list = []
    for random_seed in tqdm(range(1,11),desc="10次实验结果收集"): # tqdm(range(1,11),desc="10次实验结果收集"): # 1-10
        # acc,asr,p_num
        our_res = our_unit_res(dataset_name, model_name, attack_name, random_seed, 
                poisoned_trainset, poisoned_ids,
                filtered_poisoned_testset, clean_testset,
                device)
        asd_res = asd_unit_res(dataset_name, model_name, attack_name, random_seed, 
                poisoned_trainset, poisoned_ids,
                filtered_poisoned_testset, clean_testset,
                device)
        our_acc_list.append(our_res["acc"])
        our_asr_list.append(our_res["asr"])
        our_p_num_list.append(our_res["p_num"])

        asd_acc_list.append(asd_res["acc"])
        asd_asr_list.append(asd_res["asr"])
        asd_p_num_list.append(asd_res["p_num"])

    res_dict = {
        "our_acc_list":our_acc_list,
        "our_asr_list":our_asr_list,
        "our_p_num_list":our_p_num_list,

        "asd_acc_list":asd_acc_list,
        "asd_asr_list":asd_asr_list,
        "asd_p_num_list":asd_p_num_list,
    }

    save_dir = os.path.join(exp_save_dir, dataset_name, model_name, attack_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res_20260204.pkl" # "res_818.pkl"
    save_path = os.path.join(save_dir, save_file_name)
    joblib.dump(res_dict,save_path)
    print("结果保存在:",save_path)
    our_acc_avg, asd_acc_avg = compare_avg(our_acc_list, asd_acc_list)
    our_asr_avg, asd_asr_avg = compare_avg(our_asr_list, asd_asr_list)
    our_pNum_avg, asd_pNum_avg = compare_avg(our_p_num_list, asd_p_num_list)


    # 计算WTL
    acc_WTL_res = compare_WTL(our_acc_list, asd_acc_list, expect = "big", method="mannwhitneyu") # 越大越好
    asr_WTL_res = compare_WTL(our_asr_list, asd_asr_list, expect = "small",method="mannwhitneyu") # 越小越好
    p_num_WTL_res = compare_WTL(our_p_num_list, asd_p_num_list, expect = "small",method="mannwhitneyu") # 越小越好

    print(f"Scene:{dataset_name}|{model_name}|{attack_name}")
    print("ACC_list:")
    print(f"\tOur:{our_acc_list}")
    print(f"\tASD:{asd_acc_list}")

    print("ASR_list:")
    print(f"\tOur:{our_asr_list}")
    print(f"\tASD:{asd_asr_list}")

    print("PNUM_list:")
    print(f"\tOur:{our_p_num_list}")
    print(f"\tASD:{asd_p_num_list}")

    print(f"OurAvg: ASR:{our_asr_avg}, ACC:{our_acc_avg}, PNUM:{our_pNum_avg}")
    print(f"ASDAvg: ASR:{asd_asr_avg}, ACC:{asd_acc_avg}, PNUM:{asd_pNum_avg}")
    print(f"WTL: ASR:{asr_WTL_res}, ACC:{acc_WTL_res}, PNUM:{p_num_WTL_res}")

def look():
    save_dir = os.path.join(exp_root_dir, "Exp_Results", dataset_name, model_name, attack_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file_name = "res_818.pkl"  # res.pkl
    save_path = os.path.join(save_dir, save_file_name)
    res_dict = joblib.load(save_path)
    our_acc_list = res_dict["our_acc_list"]
    our_asr_list = res_dict["our_asr_list"]
    our_p_num_list = res_dict["our_p_num_list"]
    
    asd_acc_list = res_dict["asd_acc_list"]
    asd_asr_list = res_dict["asd_asr_list"]
    asd_p_num_list = res_dict["asd_p_num_list"]


    our_acc_avg, asd_acc_avg = compare_avg(our_acc_list, asd_acc_list)
    our_asr_avg, asd_asr_avg = compare_avg(our_asr_list, asd_asr_list)
    our_pNum_avg, asd_pNum_avg = compare_avg(our_p_num_list, asd_p_num_list)


    # 计算WTL
    acc_WTL_res = compare_WTL(our_acc_list, asd_acc_list, expect = "big", method="mannwhitneyu") # 越大越好
    asr_WTL_res = compare_WTL(our_asr_list, asd_asr_list, expect = "small", method="mannwhitneyu") # 越小越好
    p_num_WTL_res = compare_WTL(our_p_num_list, asd_p_num_list, expect = "small", method="mannwhitneyu") # 越小越好

    print(f"Scene:{dataset_name}|{model_name}|{attack_name}")
    print("ACC_list:")
    print(f"\tOur:{our_acc_list}")
    print(f"\tASD:{asd_acc_list}")

    print("ASR_list:")
    print(f"\tOur:{our_asr_list}")
    print(f"\tASD:{asd_asr_list}")

    print("PNUM_list:")
    print(f"\tOur:{our_p_num_list}")
    print(f"\tASD:{asd_p_num_list}")

    print(f"OurAvg: ASR:{our_asr_avg}, ACC:{our_acc_avg}, PNUM:{our_pNum_avg}")
    print(f"ASDAvg: ASR:{asd_asr_avg}, ACC:{asd_acc_avg}, PNUM:{asd_pNum_avg}")
    print(f"WTL: ASR:{asr_WTL_res}, ACC:{acc_WTL_res}, PNUM:{p_num_WTL_res}")

    return acc_WTL_res, asr_WTL_res, p_num_WTL_res


if __name__ == "__main__":
    
    
    

    '''
    跑单个场景pn,asr,acc WTL
    '''
    # device = torch.device("cuda:1")
    # dataset_name = "CIFAR10"
    # model_name = "ResNet18"
    # attack_name = "IAD"
    # print(dataset_name,model_name,attack_name)
    # main_scene()


    '''跑all场景pn,asr,acc WTL'''
    pid = os.getpid()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_name = "eval_ours_asd"
    exp_time = get_formattedDateTime()
    exp_save_dir = os.path.join(exp_root_dir,"Exp_Results",exp_name)
    os.makedirs(exp_save_dir,exist_ok=True)
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")

    print("pid:",pid)
    print("exp_root_dir:",exp_root_dir)
    print("exp_name:",exp_name)
    print("exp_time:",exp_time)
    print("exp_save_dir:",exp_save_dir)
    print("gpu_id:",gpu_id)
    
    for dataset_name in ["ImageNet2012_subset"]: # ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
        class_num = get_class_num(dataset_name)
        for model_name in ["ResNet18", "VGG19", "DenseNet"]:
            if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                continue
            for attack_name in ["BadNets","IAD","Refool", "WaNet"]:
                print(f"\n{dataset_name}|{model_name}|{attack_name}")
                main_scene()

    '''读取all场景pn,asr,acc WTL'''
    # device = torch.device("cuda:0")
    # acc_win_counter = 0
    # asr_win_counter = 0
    # pNum_win_counter = 0
    # total = 0
    # for dataset_name in ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
    #     class_num = get_class_num(dataset_name)
    #     for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #         if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #             continue
    #         for attack_name in ["BadNets", "IAD", "Refool", "WaNet"]:
    #             acc_res, asr_res, pNum_res = look()
    #             total += 1
    #             if acc_res == "Win":
    #                 acc_win_counter += 1
    #             if asr_res == "Win":
    #                 asr_win_counter += 1
    #             if pNum_res == "Win":
    #                 pNum_win_counter += 1
    # print(f"acc_win:{acc_win_counter}, asr_win:{asr_win_counter}, pNum_win:{pNum_win_counter}, total:{total}")
    