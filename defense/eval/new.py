import os
import joblib
import json
from utils.calcu_utils import compare_WTL,compare_avg


def read_result(dataset_name,model_name,attack_name):
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
    
    # 读取nc list
    nc_json_path = os.path.join(exp_root_dir,"Defense","NC","results.json")
    with open(nc_json_path,mode="r") as f:
        nc_res = json.load(f)
    
    nc_asr_list = []
    nc_acc_list = []
    for r_seed in range(1,11):
        asr = nc_res[dataset_name][model_name][attack_name][str(r_seed)]["asr"]
        acc = nc_res[dataset_name][model_name][attack_name][str(r_seed)]["acc"]
        nc_asr_list.append(asr)
        nc_acc_list.append(acc)


    # 读取strip
    '''
    strip_json_path = os.path.join(exp_root_dir,"Defense","Strip_hardCut","results_nocleanfinetune.json")
    with open(strip_json_path,mode="r") as f:
        strip_res = json.load(f)
    
    strip_asr_list = []
    strip_acc_list = []
    strip_pn_list = []
    for r_seed in range(1,7):
        asr = strip_res[dataset_name][model_name][attack_name][str(r_seed)]["best_asr"]
        acc = strip_res[dataset_name][model_name][attack_name][str(r_seed)]["best_acc"]
        pn = strip_res[dataset_name][model_name][attack_name][str(r_seed)]["PN"]
        strip_asr_list.append(asr)
        strip_acc_list.append(acc)
        strip_pn_list.append(pn)
    '''


    res = {
        "ours_asr_list": our_asr_list,
        "ours_acc_list": our_acc_list,
        "ours_pn_list":our_pn_list,
        "nc_asr_list": nc_asr_list,
        "nc_acc_list": nc_acc_list,
        # "strip_asr_list":strip_asr_list,
        # "strip_acc_list":strip_acc_list,
        # "strip_pn_list":strip_pn_list
    }
    return res





def one_scence(dataset_name,model_name,attack_name):

    res = read_result(dataset_name,model_name,attack_name)

    ours_asr_list = res["ours_asr_list"]
    ours_acc_list = res["ours_acc_list"]
    ours_pn_list = res["ours_pn_list"]
    nc_asr_list = res["nc_asr_list"]
    nc_acc_list = res["nc_acc_list"]
    # strip_asr_list = res["strip_asr_list"]
    # strip_acc_list = res["strip_acc_list"]
    # strip_pn_list = res["strip_pn_list"]

    print("Ours:")
    print(f"\tasr_list:{ours_asr_list}")
    print(f"\tacc_list:{ours_acc_list}")
    print(f"\tpn_list:{ours_pn_list}")

    print("NC:")
    print(f"\tasr_list:{nc_asr_list}")
    print(f"\tacc_list:{nc_acc_list}")

    # print("Strip:")
    # print(f"\tasr_list:{strip_asr_list}")
    # print(f"\tacc_list:{strip_acc_list}")
    # print(f"\tpn_list:{strip_pn_list}")


    
    print("Ours VS NC")
    asr_WTL_res = compare_WTL(ours_asr_list, nc_asr_list, expect = "small", method="mannwhitneyu") # 越小越好
    acc_WTL_res = compare_WTL(ours_acc_list, nc_acc_list, expect = "big", method="mannwhitneyu") # 越大越好

    ours_asr_avg,nc_asr_avg = compare_avg(ours_asr_list,nc_asr_list)
    ours_acc_avg,nc_acc_avg = compare_avg(ours_acc_list,nc_acc_list)
    print(f"\tasr_AVG: ours:{ours_asr_avg}, nc:{nc_asr_avg}")
    print(f"\tacc_AVG: ours:{ours_acc_avg}, nc:{nc_acc_avg}")
    print("\tasr_WTL",asr_WTL_res)
    print("\tacc_WTL",acc_WTL_res)



    
    # print("Ours VS Strip")
    # asr_WTL_res = compare_WTL(ours_asr_list, strip_asr_list, expect = "small", method="mannwhitneyu") # 越小越好
    # acc_WTL_res = compare_WTL(ours_acc_list, strip_acc_list, expect = "big", method="mannwhitneyu") # 越大越好
    # pn_WTL_res = compare_WTL(ours_pn_list, strip_pn_list, expect = "small", method="mannwhitneyu") # 越小越好

    # ours_asr_avg,strip_asr_avg = compare_avg(ours_asr_list,strip_asr_list)
    # ours_acc_avg,strip_acc_avg = compare_avg(ours_acc_list,strip_acc_list)
    # ours_pn_avg,strip_pn_avg = compare_avg(ours_pn_list,strip_pn_list)

    
    
    # print(f"\tasr_AVG: ours:{ours_asr_avg}, strip:{strip_asr_avg}")
    # print(f"\tacc_AVG: ours:{ours_acc_avg}, strip:{strip_acc_avg}")
    # print(f"\tpn_AVG: ours:{ours_pn_avg}, strip:{strip_pn_avg}")
    # print("\tasr_WTL",asr_WTL_res)
    # print("\tacc_WTL",acc_WTL_res)
    # print("\tpn_WTL",pn_WTL_res)


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
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            for attack_name in attack_name_list:
                if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                    continue
                print(f"\n{dataset_name}|{model_name}|{attack_name}")
                one_scence(dataset_name,model_name,attack_name)


