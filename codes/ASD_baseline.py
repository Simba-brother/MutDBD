import logging
import sys
from codes.utils import my_excepthook
sys.excepthook = my_excepthook
import os
import time
import torch
import torch.nn as nn
import setproctitle
from torch.utils.data import DataLoader,Dataset
from codes import config
from codes.asd import defence_train
from codes.scripts.dataset_constructor import *
from codes.common.time_handler import get_formattedDateTime
from codes.models import get_model
from codes.common.eval_model import EvalModel
from codes.utils import convert_to_hms
from codes.scripts.dataset_constructor import ExtractDataset,ExtractDataset_NormalPattern

# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_poisoned_dataset as gtsrb_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_poisoned_dataset as gtsrb_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_poisoned_dataset as gtsrb_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_poisoned_dataset as gtsrb_WaNet_gen_poisoned_dataset
# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset as imagenet_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_poisoned_dataset as imagenet_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_poisoned_dataset as imagenet_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_poisoned_dataset as imagenet_WaNet_gen_poisoned_dataset

# transform数据集
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets, gtsrb_IAD, gtsrb_Refool, gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets, imagenet_IAD, imagenet_Refool, imagenet_WaNet

# from codes.tools import model_train_test

def get_spec_dataset(dataset_name, model_name, attack_name, poisoned_ids):
    # 根据poisoned_ids得到非预制菜poisoneds_trainset
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets": # BadNets中毒操作比较快
            poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_BadNets()
        elif attack_name == "IAD": # 中毒操作较慢，而且中毒后没有数据处理步骤了
            poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = cifar10_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_WaNet()
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = gtsrb_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_WaNet()
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = imagenet_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_Refool()
        elif attack_name == "WaNet":
            # 硬盘中的数据集信息
            poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_WaNet()
    return poisoned_trainset, clean_trainset, clean_testset

def main():
    logger = get_logger()
    # 进程名称
    proctitle = f"{baseline_name}|{dataset_name}|{model_name}|{attack_name}|{rand_seed}"
    setproctitle.setproctitle(proctitle)
    logger.info(proctitle)
    logger.info(f"rand_seed:{rand_seed}")

    # 加载后门攻击配套数据

    backdoor_data = torch.load(os.path.join(
                            config.exp_root_dir, "ATTACK",
                            dataset_name,model_name,attack_name,
                            "backdoor_data.pth"), map_location="cpu")
    # backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_testset = backdoor_data["poisoned_testset"] # fixed

    victim_model = get_model(dataset_name,model_name)
    poisoned_trainset, clean_trainset, clean_testset = get_spec_dataset(
        dataset_name, model_name, attack_name, poisoned_ids)
    
    '''
    # 提前把poisoned_trainset加载到内存中
    extract_time_start = time.perf_counter()
    extracted_poisoned_trainset_1 = ExtractDataset(poisoned_trainset)
    extracted_poisoned_trainset_2 = ExtractDataset(poisoned_trainset)
    extract_time_end = time.perf_counter()
    extract_cost_seconds = extract_time_end - extract_time_start
    hours, minutes, seconds = convert_to_hms(extract_cost_seconds)
    logger.info(f"抽取2份训练集耗时:{hours}时{minutes}分{seconds:.3f}秒")'
    '''


    # dataset_list = []
    # start_time = time.perf_counter()
    # for i in range(len(extracted_poisoned_trainset_1)):
    #     x,y,flag = extracted_poisoned_trainset_1[i]
    #     dataset_list.append((x,y,flag))
    # end_time = time.perf_counter()
    # cost_time = end_time - start_time
    # hours, minutes, seconds = convert_to_hms(cost_time)
    # print(f"遍历被提前抽取耗时:{hours}时{minutes}分{seconds:.3f}秒")



    # loder_time_start = time.perf_counter()
    # loder = DataLoader(poisoned_trainset, batch_size=64, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)
    # for _, batch in enumerate(loder):
    #     X = batch[0]
    #     # 该批次标签
    #     Y = batch[1]
    # loder_time_end = time.perf_counter()
    # loader_cost_time = loder_time_end - loder_time_start
    # hours, minutes, seconds = convert_to_hms(loader_cost_time)
    # print(f"loader遍历一遍新鲜训练集耗时:{hours}时{minutes}分{seconds:.3f}秒")


    # loder_time_start = time.perf_counter()
    # loder = DataLoader(extracted_poisoned_trainset_1, batch_size=64, num_workers=0, pin_memory=True, shuffle=True, drop_last=True)
    # for _, batch in enumerate(loder):
    #     X = batch[0]
    #     # 该批次标签
    #     Y = batch[1]
    # loder_time_end = time.perf_counter()
    # loader_cost_time = loder_time_end - loder_time_start
    # hours, minutes, seconds = convert_to_hms(loader_cost_time)
    # print(f"loader遍历一遍提前抽取训练集耗时:{hours}时{minutes}分{seconds:.3f}秒")

    # def PreloadedDataset(Dataset):
    #     def __init__(self,data,labels):
    #         self.data = data
    #         self.labels = labels
    #         self.is_p = is_p_list
    #     def __len__(self):
    #         return len(self.data)
    #     def __getitem__(self,idx):
    #         return self.data[idx],self.labels[idx]


    # extract_time_start = time.perf_counter()
    # extracted_clean_trainset = ExtractDataset_NormalPattern(clean_trainset)
    # extract_time_end = time.perf_counter()
    # extract_cost_seconds = extract_time_end - extract_time_start
    # hours,minutes,seconds = convert_to_hms(extract_cost_seconds)
    # print(f"抽取干净训练集耗时:{hours}时{minutes}分{seconds:.3f}秒")

    # # 提前抽取之前遍历一遍数据集耗时
    # start_time = time.perf_counter()
    # for i in range(len(poisoned_trainset)):
    #     x,y,flag = poisoned_trainset[i]
    # end_time = time.perf_counter()
    # cost_time = end_time - start_time
    # hours, minutes, seconds = convert_to_hms(cost_time)
    # print(f"遍历新鲜耗时:{hours}时{minutes}分{seconds:.3f}秒")


    # extracted_poisoned_trainset_1_loader = DataLoader(
    #             extracted_poisoned_trainset_1, # 新鲜
    #             batch_size=64,
    #             shuffle=True,
    #             num_workers=4,
    #             pin_memory=True)

    # extracted_poisoned_trainset_2_loader = DataLoader(
    #             extracted_poisoned_trainset_2, # 
    #             batch_size=64,
    #             shuffle=True,
    #             num_workers=4,
    #             pin_memory=True)

    # extracted_poisoned_evalset_loader = DataLoader(
    #             extracted_poisoned_trainset_1, # 
    #             batch_size=64,
    #             shuffle=False,
    #             num_workers=4,
    #             pin_memory=True)


    # 数据加载器
    poisoned_trainset_loader = DataLoader(
                poisoned_trainset, # 新鲜
                batch_size=64,
                shuffle=True,
                num_workers=4,
                pin_memory=True)

    poisoned_evalset_loader = DataLoader(
                poisoned_trainset, # 新鲜
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)

    clean_testset_loader = DataLoader(
                clean_testset, # 新鲜
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)

    poisoned_testset_loader = DataLoader(
                poisoned_testset, # 预制
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    # 获得设备
    device = torch.device(f"cuda:{gpu_id}")

    # 开始防御式训练
    logger.info("开始ASD防御式训练")
    time_1 = time.perf_counter()
    exp_dir= os.path.join(
                    config.exp_root_dir, 
                    f"{baseline_name}", 
                    dataset_name, 
                    model_name, 
                    attack_name, 
                    time.strftime("%Y-%m-%d_%H:%M:%S")
                    )
    logger.info(f"实验目录:{exp_dir}")
    best_ckpt_path, latest_ckpt_path = defence_train(
            model = victim_model, # victim model
            class_num = class_num, # 分类数量
            poisoned_train_dataset = poisoned_trainset,
            poisoned_ids = poisoned_ids, # 被污染的样本id list
            poisoned_eval_dataset_loader = poisoned_evalset_loader, # （新鲜）有污染的训练集加载器（不打乱加载）
            poisoned_train_dataset_loader = poisoned_trainset_loader, # （新鲜）有污染的训练集加载器（打乱加载）
            clean_test_dataset_loader = clean_testset_loader, # 干净的测试集加载器
            poisoned_test_dataset_loader = poisoned_testset_loader, # 污染的测试集加载器（预制）
            device=device, # GPU设备对象
            # 实验结果存储目录
            save_dir = exp_dir,
            logger=logger,
            dataset_name = dataset_name,
            model_name = model_name,
            rand_seed = rand_seed,
            # extracted_poisoned_trainset_1 = extracted_poisoned_trainset_1,
            # extracted_poisoned_trainset_2 = extracted_poisoned_trainset_2
            )
    time_2 = time.perf_counter()
    cost_time = time_2 - time_1
    hours, minutes, seconds = convert_to_hms(cost_time)
    logger.info(f"防御式训练完成，共耗时:{hours}时{minutes}分{seconds:.3f}秒")
    # 评估防御结果
    logger.info("开始评估防御结果")

    best_model_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    victim_model.load_state_dict(best_model_ckpt["model_state_dict"])
    new_model = victim_model
    # (1) 评估新模型在clean testset上的acc
    em = EvalModel(new_model,clean_testset,device)
    clean_test_acc = em.eval_acc()
    # (2) 评估新模型在poisoned testset上的acc
    em = EvalModel(new_model,poisoned_testset,device)
    poisoned_test_acc = em.eval_acc()
    logger.info(f"BestModel: ACC:{clean_test_acc}, ASR:{poisoned_test_acc}")

    last_model_ckpt = torch.load(latest_ckpt_path, map_location="cpu")
    victim_model.load_state_dict(last_model_ckpt["model_state_dict"])
    new_model = victim_model
    # (1) 评估新模型在clean testset上的acc
    em = EvalModel(new_model,clean_testset,device)
    clean_test_acc = em.eval_acc()
    # (2) 评估新模型在poisoned testset上的acc
    em = EvalModel(new_model,poisoned_testset,device)
    poisoned_test_acc = em.eval_acc()
    logger.info(f"LastModel: ACC:{clean_test_acc}, ASR:{poisoned_test_acc}")


    time_4 = time.perf_counter()
    cost_time = time_4 - time_2
    hours, minutes, seconds = convert_to_hms(cost_time)
    logger.info(f"评估防御结果结束，共耗时:{hours}时{minutes}分{seconds:.3f}秒")



def _get_logger(log_dir,log_file_name,logger_name):
    # 创建一个logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,log_file_name)

    # logger的文件处理器，包括日志等级，日志路径，模式，编码等
    file_handler = logging.FileHandler(log_path,mode="w",encoding="UTF-8")
    file_handler.setLevel(logging.DEBUG)

    # logger的格式化处理器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    #将格式化器添加到文件处理器
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志对象中
    logger.addHandler(file_handler)
    return logger

def get_logger():
    # log_base_dir = "log/temp/"
    log_base_dir = f"log/{baseline_name}/"
    # 获得实验时间戳年月日时分秒
    _time = get_formattedDateTime()
    log_dir = os.path.join(log_base_dir,dataset_name,model_name,attack_name)
    log_file_name = f"retrain_r_seed_{rand_seed}_{_time}.log"
    logger = _get_logger(log_dir,log_file_name,logger_name=_time)
    return logger


def get_classNum(dataset_name):
    class_num = None
    if dataset_name == "CIFAR10":
        class_num = 10
    elif dataset_name == "GTSRB":
        class_num = 43
    elif dataset_name == "ImageNet2012_subset":
        class_num = 30
    return class_num

if __name__ == "__main__":

    # gpu_id = 1
    # rand_seed = 1
    # baseline_name = "ASD_new"
    # dataset_name= "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
    # model_name= "ResNet18" # ResNet18, VGG19, DenseNet
    # attack_name ="BadNets" # BadNets, IAD, Refool, WaNet
    # class_num = get_classNum(dataset_name)
    # main()

    gpu_id = 1
    baseline_name = "ASD_new"
    rand_seed = 1
    dataset_name = "ImageNet2012_subset"
    class_num = get_classNum(dataset_name)
    model_name = "ResNet18"
    for attack_name in ["BadNets", "IAD", "Refool", "WaNet"]:
        main()

    # for rand_seed in [10]:
    #     for dataset_name in ["CIFAR10"]: # ["CIFAR10", "GTSRB", "ImageNet2012_subset"]:
    #         if dataset_name == "CIFAR10":
    #             class_num = 10
    #         elif dataset_name == "GTSRB":
    #             class_num = 43
    #         else:
    #             class_num = 30
    #         for model_name in ["ResNet18", "VGG19", "DenseNet"]:
    #             if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
    #                 continue
    #             for attack_name in ["BadNets", "IAD", "Refool", "WaNet"]:
    #                 main()