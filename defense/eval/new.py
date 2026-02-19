import os
import joblib
import json
from utils.calcu_utils import compare_WTL,compare_avg
import pprint



def read_ours_result(dataset_name,model_name,attack_name):
    # 读取我们的list
    if dataset_name in ["CIFAR10","GTSRB"]:
        res = joblib.load(os.path.join(exp_root_dir,"Exp_Results","eval_ours_asd",
                                       dataset_name,model_name,attack_name,"res_818.pkl"))
        our_asr_list = res["our_asr_list"]
        our_acc_list = res["our_acc_list"]
        our_pn_list = res["our_p_num_list"]
    else:
        json_path = os.path.join(exp_root_dir,"Exp_Results","eval_ours_asd","ImageNet","res.json")
        with open(json_path,mode="r") as f:
            res = json.load(f)
            our_asr_list = res[dataset_name][model_name][attack_name]["ours"]["asr_list"]
            our_acc_list = res[dataset_name][model_name][attack_name]["ours"]["acc_list"]
            our_pn_list = res[dataset_name][model_name][attack_name]["ours"]["pnum_list"]

    return our_asr_list, our_acc_list, our_pn_list

def read_nc_result(dataset_name,model_name,attack_name):
    # 读取nc list
    nc_json_path = os.path.join(exp_root_dir,"Defense","NC", "run_1", "results.json")
    with open(nc_json_path,mode="r") as f:
        nc_res = json.load(f)
    
    nc_asr_list = []
    nc_acc_list = []
    for r_seed in range(1,11):
        asr = nc_res[dataset_name][model_name][attack_name][str(r_seed)]["asr"]
        acc = nc_res[dataset_name][model_name][attack_name][str(r_seed)]["acc"]
        nc_asr_list.append(asr)
        nc_acc_list.append(acc)
    return nc_asr_list, nc_acc_list

def read_strip_result(dataset_name,model_name,attack_name):
    # 读取strip
    '''
    cifa10_gtsrb_json_path = os.path.join(exp_root_dir,"Defense","Strip_hardCut",
                                        "defenseTrain", "results_2026-02-11_10:51:06.json")
    with open(cifa10_gtsrb_json_path,mode="r") as f:
        cifa10_gtsrb_res = json.load(f)
    
    imagenet_json_path = os.path.join(exp_root_dir,"Defense","Strip_hardCut",
                                        "defenseTrain", "results_2026-02-11_10:52:01.json")
    with open(imagenet_json_path,mode="r") as f:
        imagenet_res = json.load(f)
    '''
    json_path = os.path.join(exp_root_dir,"Defense","Strip_hardCut",
                                        "defenseTrain", "results.json")
    with open(json_path,mode="r") as f:
        res = json.load(f)
    
    strip_asr_list = []
    strip_acc_list = []
    strip_pn_list = []


    for r_seed in range(1,11):
        asr = res[dataset_name][model_name][attack_name][str(r_seed)]["best_asr"]
        acc = res[dataset_name][model_name][attack_name][str(r_seed)]["best_acc"]
        pn = res[dataset_name][model_name][attack_name][str(r_seed)]["PN"]
        '''
        if dataset_name in ["CIFAR10","GTSRB"]:
            asr = cifa10_gtsrb_res[dataset_name][model_name][attack_name][str(r_seed)]["best_asr"]
            acc = cifa10_gtsrb_res[dataset_name][model_name][attack_name][str(r_seed)]["best_acc"]
            pn = cifa10_gtsrb_res[dataset_name][model_name][attack_name][str(r_seed)]["PN"]
        elif dataset_name in ["ImageNet2012_subset"] and r_seed <= 4:
            asr = imagenet_res[dataset_name][model_name][attack_name][str(r_seed)]["best_asr"]
            acc = imagenet_res[dataset_name][model_name][attack_name][str(r_seed)]["best_acc"]
            pn = imagenet_res[dataset_name][model_name][attack_name][str(r_seed)]["PN"]
        '''
        strip_asr_list.append(asr)
        strip_acc_list.append(acc)
        strip_pn_list.append(pn)
    return strip_asr_list, strip_acc_list, strip_pn_list
    
    


def read_result(dataset_name,model_name,attack_name):
    
    our_asr_list, our_acc_list, our_pn_list = read_ours_result(dataset_name,model_name,attack_name)
    nc_asr_list, nc_acc_list = read_nc_result(dataset_name,model_name,attack_name)
    strip_asr_list, strip_acc_list, strip_pn_list = read_strip_result(dataset_name,model_name,attack_name)

    res = {
        "ours_asr_list": our_asr_list,
        "ours_acc_list": our_acc_list,
        "ours_pn_list":our_pn_list,

        "nc_asr_list": nc_asr_list,
        "nc_acc_list": nc_acc_list,

        "strip_asr_list": strip_asr_list,
        "strip_acc_list": strip_acc_list,
        "strip_pn_list": strip_pn_list,

    }
    return res



def one_scence_ours_vs_strip(dataset_name,model_name,attack_name):
    ours_asr_list, ours_acc_list, ours_pn_list = read_ours_result(dataset_name,model_name,attack_name)
    strip_asr_list, strip_acc_list, strip_pn_list = read_strip_result(dataset_name,model_name,attack_name)
    print("Ours:")
    print(f"\tasr_list:{ours_asr_list}")
    print(f"\tacc_list:{ours_acc_list}")
    print(f"\tpn_list:{ours_pn_list}")

    print("Strip:")
    print(f"\tasr_list:{strip_asr_list}")
    print(f"\tacc_list:{strip_acc_list}")

    
    print("Ours vs. Strip")
    asr_WTL_res = compare_WTL(ours_asr_list, strip_asr_list, expect = "small", method="mannwhitneyu") # 越小越好
    acc_WTL_res = compare_WTL(ours_acc_list, strip_acc_list, expect = "big", method="mannwhitneyu") # 越大越好

    ours_asr_avg,strip_asr_avg = compare_avg(ours_asr_list,strip_asr_list)
    ours_acc_avg,strip_acc_avg = compare_avg(ours_acc_list,strip_acc_list)

    asr_avg_flag = "Lose"
    if ours_asr_avg < strip_asr_avg:
        asr_avg_flag = "Win"
    acc_avg_flag = "Lose"
    if ours_acc_avg > strip_acc_avg:
        acc_avg_flag = "Win"

    res = {
        "ASR":{
            "ours":ours_asr_avg,
            "strip":strip_asr_avg,
            "AVG_flag": asr_avg_flag,
            "W/T/L": asr_WTL_res
        },
        "ACC":{
            "ours":ours_acc_avg,
            "strip":strip_acc_avg,
            "AVG_flag":acc_avg_flag,
            "W/T/L":acc_WTL_res,
        }
    }
    return res




def one_scence_ours_vs_nc(dataset_name,model_name,attack_name):
    ours_asr_list, ours_acc_list, ours_pn_list = read_ours_result(dataset_name,model_name,attack_name)
    nc_asr_list, nc_acc_list = read_nc_result(dataset_name,model_name,attack_name)
    print("Ours:")
    print(f"\tasr_list:{ours_asr_list}")
    print(f"\tacc_list:{ours_acc_list}")
    print(f"\tpn_list:{ours_pn_list}")

    print("NC:")
    print(f"\tasr_list:{nc_asr_list}")
    print(f"\tacc_list:{nc_acc_list}")

    
    print("Ours vs. NC")
    asr_WTL_res = compare_WTL(ours_asr_list, nc_asr_list, expect = "small", method="mannwhitneyu") # 越小越好
    acc_WTL_res = compare_WTL(ours_acc_list, nc_acc_list, expect = "big", method="mannwhitneyu") # 越大越好

    ours_asr_avg,nc_asr_avg = compare_avg(ours_asr_list,nc_asr_list)
    ours_acc_avg,nc_acc_avg = compare_avg(ours_acc_list,nc_acc_list)

    asr_avg_flag = "Lose"
    if ours_asr_avg < nc_asr_avg:
        asr_avg_flag = "Win"
    acc_avg_flag = "Lose"
    if ours_acc_avg > nc_acc_avg:
        acc_avg_flag = "Win"

    res = {
        "ASR":{
            "ours":ours_asr_avg,
            "nc":nc_asr_avg,
            "AVG_flag": asr_avg_flag,
            "W/T/L": asr_WTL_res
        },
        "ACC":{
            "ours":ours_acc_avg,
            "nc":nc_acc_avg,
            "AVG_flag":acc_avg_flag,
            "W/T/L":acc_WTL_res,
        }
    }
    return res


def total_count(all_scence_res):
    count_result = {
        "ASR":{
            "AVG":{
                "Win":0,
                "Lose":0
            },
            "WTL":{
                "Win":0,
                "Tie":0,
                "Lose":0
            }
        },
        "ACC":{
            "AVG":{
                "Win":0,
                "Lose":0
            },
            "WTL":{
                "Win":0,
                "Tie":0,
                "Lose":0
            }
        }
    }
    for res in all_scence_res:
        if res["ASR"]["AVG_flag"] == "Win":
            count_result["ASR"]["AVG"]["Win"] += 1
        else:
            count_result["ASR"]["AVG"]["Lose"] += 1
        
        if res["ASR"]["W/T/L"] == "Win":
            count_result["ASR"]["WTL"]["Win"] += 1
        elif res["ASR"]["W/T/L"] == "Tie":
            count_result["ASR"]["WTL"]["Tie"] += 1
        elif res["ASR"]["W/T/L"] == "Lose":
            count_result["ASR"]["WTL"]["Lose"] += 1

        if res["ACC"]["AVG_flag"] == "Win":
            count_result["ACC"]["AVG"]["Win"] += 1
        else:
            count_result["ACC"]["AVG"]["Lose"] += 1
        
        if res["ACC"]["W/T/L"] == "Win":
            count_result["ACC"]["WTL"]["Win"] += 1
        elif res["ACC"]["W/T/L"] == "Tie":
            count_result["ACC"]["WTL"]["Tie"] += 1
        elif res["ACC"]["W/T/L"] == "Lose":
            count_result["ACC"]["WTL"]["Lose"] += 1

    return count_result



if __name__ == "__main__":
    # one_scence
    '''
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    print(f"\n{dataset_name}|{model_name}|{attack_name}")
    one_scence(dataset_name,model_name,attack_name)
    '''

    # all_scence
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]

    all_scence_res = []
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                print(f"\n{dataset_name}|{model_name}|{attack_name}")
                res = one_scence_ours_vs_strip(dataset_name,model_name,attack_name)
                pprint.pprint(res)
                all_scence_res.append(res)

    count_result = total_count(all_scence_res)
    print("\n统计结果:")
    pprint.pprint(count_result)



