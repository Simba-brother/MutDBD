'''
生成所有变异模型的评估结果
'''
import os
import time
import copy
import torch
import logging
import setproctitle
import pandas as pd
import joblib
from utils.model_eval_utils import EvalModel
from datasets.utils import ExtractDataset_NoPoisonedFlag
from datasets.posisoned_dataset import get_all_dataset,get_clean_dataset
from models.model_loader import get_model
from utils.common_utils import convert_to_hms

def get_mutation_models_pred_labels(dataset):
    mutations_dir = os.path.join(
        exp_root_dir,
        "MutationModels", # "MutationsForDiscussion", # MutationModels
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
    mutation_model_num = 100
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



def main_one_scence(exp_sub,save_format):
    if exp_sub == "preLabel": # 主
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
        print(f"变异率 {rate*100}% 完成")
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
            print(f"csv保存在:{save_file_path}")
        elif save_format == "joblib":
            save_file_name = f"{exp_sub}.joblib"
            save_file_path = os.path.join(save_dir,save_file_name)
            ansTodictData(data_dict,save_file_path)
            print(f"字典数据保存在:{save_file_path}")
        elif save_format == "parquet":
            save_file_name = f"{exp_sub}.parquet"
            save_file_path = os.path.join(save_dir,save_file_name)
            print(data_dict,save_file_path)
        

if __name__ == "__main__":
    # one-scence
    '''
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10" # CIFAR10,GTSRB,ImageNet2012_subset
    model_name = "ResNet18" # ResNet18,VGG19,DenseNet
    attack_name = "BadNets" # BadNets,IAD,Refool,WaNet,LabelConsistent
    gpu_id = 1
    main_exp_name = "EvalMutationToCSV" # "EvalMutationToCSV_ForDiscussion" 
    sub_exp_name = "preLabel"
    mutation_rate_list = [0.001] # [0.01] # [0.01, 0.03, 0.05, 0.07, 0.09, 0.1] # [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] 
    mutation_name_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
    mutation_model_num = 100 # 每个变异算子生成了100个模型，总共5个变异算子，500个。
    # 进程名称
    proctitle = f"{main_exp_name}|{dataset_name}|{model_name}|{attack_name}"
    setproctitle.setproctitle(proctitle)
    device = torch.device(f"cuda:{gpu_id}")

    # 日志保存目录
    
    # LOG_FORMAT = "时间：%(asctime)s - 日志等级：%(levelname)s - 日志信息：%(message)s"
    # LOG_FILE_DIR = os.path.join("log","EvalMutationToCSV",dataset_name,model_name,attack_name)
    # os.makedirs(LOG_FILE_DIR,exist_ok=True)
    # LOG_FILE_NAME = f"{main_exp_name}_{sub_exp_name}.log"
    # LOG_FILE_PATH = os.path.join(LOG_FILE_DIR,LOG_FILE_NAME)
    # logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT,filename=LOG_FILE_PATH,filemode="w")
    # logging.debug(proctitle)
    

    try:
        print(f"开始:加载后门模型数据")
        backdoor_data_path = os.path.join(
            exp_root_dir, 
            "ATTACK", 
            dataset_name, 
            model_name, 
            attack_name, 
            "backdoor_data.pth")
        backdoor_data = torch.load(backdoor_data_path, map_location="cpu")

        if "backdoor_model" in backdoor_data.keys():
            backdoor_model = backdoor_data["backdoor_model"]
        else:
            model = get_model(dataset_name, model_name)
            state_dict = backdoor_data["backdoor_model_weights"]
            model.load_state_dict(state_dict)
            backdoor_model = model
       
        backdoor_model_origin = copy.deepcopy(backdoor_model)
        poisoned_ids = backdoor_data["poisoned_ids"]
        if dataset_name == "CIFAR10":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
            poisoned_trainset = ExtractDataset_NoPoisonedFlag(poisoned_trainset)
        elif dataset_name == "GTSRB":
            poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset, new_poisoned_ids,adv_asr = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
            print(f"对抗攻击成功率:{adv_asr}")
            # 花点时间预抽取下中毒训练集，为了加速评估
            poisoned_trainset = ExtractDataset_NoPoisonedFlag(poisoned_trainset)
        print(f"开始:得到所有变异模型在poisoned trainset上的预测{sub_exp_name}结果")
        main(f"{sub_exp_name}",save_format="csv")
        print("End")
    except Exception as e:
        print("发生异常:%s",e)
    '''

    # all-scence
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10"] # ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18"] # ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["SBA"] # ["BadNets","IAD","Refool","WaNet"]
    gpu_id = 1

    mutation_name_list = ["Gaussian_Fuzzing","Weight_Shuffling","Neuron_Activation_Inverse","Neuron_Block","Neuron_Switch"]
    mutation_rate_list = [0.01] # [0.007] # [0.01] # [0.01, 0.03, 0.05, 0.07, 0.09, 0.1] # [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] 
    mutation_model_num = 100 # 每个变异算子生成了100个模型，总共5个变异算子，500个。

    main_exp_name = "EvalMutationToCSV" # "EvalMutationToCSV_ForDiscussion" 
    sub_exp_name = "preLabel"

    print(exp_root_dir)
    print(dataset_name_list)
    print(model_name_list)
    print(attack_name_list)
    print(mutation_name_list)
    print(mutation_rate_list)
    print(mutation_model_num)
    print(main_exp_name)
    print(sub_exp_name)
    all_start_time = time.perf_counter()
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                print("==="*30)
                print(f"{main_exp_name}|{dataset_name}|{model_name}|{attack_name}")
                # 进程名称
                proctitle = f"{main_exp_name}|{dataset_name}|{model_name}|{attack_name}"
                setproctitle.setproctitle(proctitle)
                # 声明gpu设备
                device = torch.device(f"cuda:{gpu_id}")    
                try:
                    print(f"开始:加载后门模型数据")
                    backdoor_data_path = os.path.join(
                        exp_root_dir, 
                        "ATTACK", 
                        dataset_name, 
                        model_name, 
                        attack_name, 
                        "backdoor_data.pth")
                    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")

                    if "backdoor_model" in backdoor_data.keys():
                        backdoor_model = backdoor_data["backdoor_model"]
                    else:
                        model = get_model(dataset_name, model_name)
                        state_dict = backdoor_data["backdoor_model_weights"]
                        model.load_state_dict(state_dict)
                        backdoor_model = model
                
                    backdoor_model_origin = copy.deepcopy(backdoor_model)
                    if attack_name == "SBA":
                        trainset, testset = get_clean_dataset(dataset_name,attack_name)
                        poisoned_trainset = trainset
                        poisoned_ids = []
                    else:
                        poisoned_ids = backdoor_data["poisoned_ids"]
                        poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
                        poisoned_trainset = ExtractDataset_NoPoisonedFlag(poisoned_trainset)

                    print(f"开始:得到所有变异模型在poisoned trainset上的预测{sub_exp_name}结果")
                    one_scence_start_time = time.perf_counter()
                    main_one_scence(f"{sub_exp_name}",save_format="csv")
                    one_scence_end_time = time.perf_counter()
                    one_scence_cost_time = one_scence_end_time - one_scence_start_time
                    hours, minutes, seconds = convert_to_hms(one_scence_cost_time)
                    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
                    # if dataset_name == "CIFAR10":
                    #     poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
                    #     poisoned_trainset = ExtractDataset_NoPoisonedFlag(poisoned_trainset)
                    # elif dataset_name == "GTSRB":
                    #     poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset, new_poisoned_ids,adv_asr = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
                    #     print(f"对抗攻击成功率:{adv_asr}")
                    #     # 花点时间预抽取下中毒训练集，为了加速评估
                    #     poisoned_trainset = ExtractDataset_NoPoisonedFlag(poisoned_trainset)
                except Exception as e:
                    print("发生异常:%s",e)
    all_end_time = time.perf_counter()
    all_cost_time = all_end_time - all_start_time
    hours, minutes, seconds = convert_to_hms(all_cost_time)
    print(f"all-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
