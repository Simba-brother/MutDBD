'''
重要
使用变异模型对训练集进行评估
'''
import os

import torch
import logging
import setproctitle
import pandas as pd
import joblib

from codes.scripts.dataset_constructor import *
from codes.common.eval_model import EvalModel
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



def get_mutation_models_pred_labels(dataset):
    mutations_dir = os.path.join(
        exp_root_dir,
        "MutationsForDiscussion", # MutationModels
        dataset_name,
        model_name,
        attack_name
    )
    eval_ans = {}
    for ratio in mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device,batch_size=512, num_workers=8)
                pred_label_list = em.get_pred_labels()
                eval_ans[ratio][operator].append(pred_label_list)
    return eval_ans

def get_mutation_models_CELoss(dataset):
    mutations_dir = os.path.join(
        exp_root_dir,
        "MutationModels",
        dataset_name,
        model_name,
        attack_name
    )
    em = EvalModel(backdoor_model, dataset, device, batch_size=512, num_workers=8)
    eval_ans = {}
    for ratio in mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(mutation_model_num):
                logging.debug(f"\t\tmodel_id:{i}")
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em.model = backdoor_model
                CELoss_list = em.get_CEloss()
                eval_ans[ratio][operator].append(CELoss_list)
    return eval_ans

def get_mutation_models_confidence(dataset):
    mutations_dir = os.path.join(
        exp_root_dir,
        "MutationModels",
        dataset_name,
        model_name,
        attack_name
    )
    eval_ans = {}
    for ratio in mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(mutation_model_num):
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em = EvalModel(backdoor_model,dataset,device, batch_size=512, num_workers=8)
                confidence_list = em.get_confidence_list()
                eval_ans[ratio][operator].append(confidence_list)
    return eval_ans

def get_mutation_models_prob_outputs(dataset):
    ''''
    得到变异模型的概率输出
    '''
    mutations_dir = os.path.join(
        exp_root_dir,
        "MutationModels",
        dataset_name,
        model_name,
        attack_name
    )
    em = EvalModel(backdoor_model, dataset, device, batch_size=512, num_workers=8)
    '''
    {rate:operator:[prob_outputs]}
    '''
    eval_ans = {}
    # 遍历变异率
    for ratio in mutation_rate_list:
        logging.debug(f"变异率:{str(ratio)}")
        eval_ans[ratio] = {}
        for operator in mutation_name_list:
            logging.debug(f"\t变异算子:{operator}")
            eval_ans[ratio][operator] = []
            for i in range(mutation_model_num):
                logging.debug(f"\t\t变异model_id:{i}")
                mutation_model_path = os.path.join(mutations_dir,str(ratio),operator,f"model_{i}.pth")
                backdoor_model.load_state_dict(torch.load(mutation_model_path))
                em.model = backdoor_model
                prob_outputs_list = em.get_prob_outputs()
                eval_ans[ratio][operator].append(prob_outputs_list)
    return eval_ans


def ansToCSV(data_dict,original_backdoorModel_preLabel_list,save_path):
    # 所有的变异模型列
    # global_model_id：[0-99]是GF变异模型，[100-199]是WS变异模型，
    # [200-299]是NAI变异模型，[300-399]是NB变异模型，[400-499]是NS变异模型
    mutation_operator_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
    mutation_model_num = 10
    total_dict = {}
    global_model_id = 0
    for operator in mutation_operator_list:
        for i in range(mutation_model_num):
            pred_label_list = data_dict[operator][i]
            model_name = f"model_{global_model_id}"
            total_dict[model_name] = pred_label_list # 变异模型列
            global_model_id += 1
    # sampled_id列,GT_label列和isPoisoned列
    total_dict["original_backdoorModel_preLabel"] = original_backdoorModel_preLabel_list
    total_dict["sampled_id"] = []
    total_dict["GT_label"] = []
    total_dict["isPoisoned"] = []
    for i in range(len(poisoned_trainset)):
        total_dict["sampled_id"].append(i)
        total_dict["GT_label"].append(poisoned_trainset[i][1])
        if i in poisoned_ids:
            total_dict["isPoisoned"].append(True)
        else:
            total_dict["isPoisoned"].append(False)
    # 构建DataFrame
    df = pd.DataFrame(total_dict)
    # 保存为csv
    df.to_csv(save_path,index=False)


def ansToParquet(data_dict,save_path):
    # 所有的变异模型列
    # global_model_id：[0-99]是GF变异模型，[100-199]是WS变异模型，
    # [200-299]是NAI变异模型，[300-399]是NB变异模型，[400-499]是NS变异模型
    total_dict = {}
    global_model_id = 0
    for operator in mutation_name_list:
        for i in range(mutation_model_num):
            pred_label_list = data_dict[operator][i]
            model_name = f"model_{global_model_id}"
            total_dict[model_name] = pred_label_list # 变异模型列
            global_model_id += 1

    # sampled_id列,GT_label列和isPoisoned列
    total_dict["sampled_id"] = []
    total_dict["GT_label"] = []
    total_dict["isPoisoned"] = []
    for i in range(len(poisoned_trainset)):
        total_dict["sampled_id"].append(i)
        total_dict["GT_label"].append(poisoned_trainset[i][1])
        if i in poisoned_ids:
            total_dict["isPoisoned"].append(True)
        else:
            total_dict["isPoisoned"].append(False)
    # 构建DataFrame
    df = pd.DataFrame(total_dict)
    # 保存为csv
    df.to_parquet(save_path,index=False)

def ansTodictData(data_dict,save_path):
    total_dict = {}
    global_model_id = 0
    for operator in mutation_name_list:
        for i in range(mutation_model_num):
            pred_label_list = data_dict[operator][i]
            model_name = f"model_{global_model_id}"
            total_dict[model_name] = pred_label_list # 变异模型列
            global_model_id += 1
    # sampled_id列,GT_label列和isPoisoned列
    total_dict["sampled_id"] = []
    total_dict["GT_label"] = []
    total_dict["isPoisoned"] = []
    for i in range(len(poisoned_trainset)):
        total_dict["sampled_id"].append(i)
        total_dict["GT_label"].append(poisoned_trainset[i][1])
        if i in poisoned_ids:
            total_dict["isPoisoned"].append(True)
        else:
            total_dict["isPoisoned"].append(False)
    joblib.dump(total_dict,save_path)



def main(exp_sub,save_format):
    if exp_sub == "preLabel":
        em = EvalModel(backdoor_model_origin, poisoned_trainset, device)
        original_backdoorModel_preLabel_list = em.get_pred_labels()
        mutation_models_pred_dict = get_mutation_models_pred_labels(poisoned_trainset)
    elif exp_sub == "confidence":
        mutation_models_pred_dict = get_mutation_models_confidence(poisoned_trainset)
    elif exp_sub == "CELoss":
        mutation_models_pred_dict = get_mutation_models_CELoss(poisoned_trainset)
    elif exp_sub == "prob_outputs":
        mutation_models_pred_dict = get_mutation_models_prob_outputs(poisoned_trainset)
    logging.debug(f"开始:将结果整理为csv文件")
    for rate in mutation_rate_list:
        data_dict = mutation_models_pred_dict[rate]
        save_dir = os.path.join(
            exp_root_dir,
            main_exp_name,
            dataset_name,
            model_name,
            attack_name,
            str(rate)
        )
        os.makedirs(save_dir,exist_ok=True)
        if save_format == "csv":
            save_file_name = f"{exp_sub}.csv"
            save_file_path = os.path.join(save_dir,save_file_name)
            ansToCSV(data_dict,original_backdoorModel_preLabel_list,save_file_path)
            logging.debug(f"csv保存在:{save_file_path}")
        elif save_format == "joblib":
            save_file_name = f"{exp_sub}.joblib"
            save_file_path = os.path.join(save_dir,save_file_name)
            ansTodictData(data_dict,save_file_path)
            logging.debug(f"字典数据保存在:{save_file_path}")
        elif save_format == "parquet":
            save_file_name = f"{exp_sub}.parquet"
            save_file_path = os.path.join(save_dir,save_file_name)
            ansToParquet(data_dict,save_file_path)


def get_poisoned_trainset(poisoned_ids):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(model_name,poisoned_ids,"train")
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "IAD":
            poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
        elif attack_name == "Refool":
            poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
        elif attack_name == "WaNet":
            poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
    return poisoned_trainset


if __name__ == "__main__":
    # 进程名称
    exp_root_dir = "/data/mml/backdoor_detect/experiments/"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    main_exp_name = "EvalMutationToCSV_ForDiscussion" 
    sub_exp_name = "preLabel"
    mutation_rate_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1] # [0.03, 0.05, 0.07, 0.09, 0.1]
    mutation_name_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
    mutation_model_num = 10
    proctitle = f"{main_exp_name}|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cuda:1")

    # 日志保存目录
    LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    LOG_FILE_DIR = os.path.join("log","EvalMutationToCSV_ForDiscussion",dataset_name,model_name,attack_name)
    os.makedirs(LOG_FILE_DIR,exist_ok=True)
    LOG_FILE_NAME = f"{main_exp_name}_{sub_exp_name}.log"
    LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    logging.debug(proctitle)

    try:
        logging.debug(f"开始:加载后门模型数据")
        backdoor_data_path = os.path.join(
            exp_root_dir, 
            "ATTACK", 
            dataset_name, 
            model_name, 
            attack_name, 
            "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
        backdoor_model = backdoor_data["backdoor_model"]
        backdoor_model_origin = copy.deepcopy(backdoor_model)
        poisoned_ids = backdoor_data["poisoned_ids"]
        poisoned_trainset = get_poisoned_trainset(poisoned_ids)
        logging.debug(f"开始:得到所有变异模型在poisoned trainset上的预测{sub_exp_name}结果")
        main(f"{sub_exp_name}",save_format="csv")
        logging.debug("End")
    except Exception as e:
        logging.error("发生异常:%s",e)





