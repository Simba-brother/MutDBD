
import os
import json
from collections import defaultdict

def load_data(results_json_path):
    with open(results_json_path, mode="r") as f:
        results = json.load(f)
    data = {}
    for attack_name in attack_name_list:
        data[attack_name] = defaultdict(list)
        for r_seed in r_seed_list:
            res = results[dataset_name][model_name][attack_name][str(r_seed)]
            data[attack_name]["pn"].append(res["PN"])
            data[attack_name]["asr"].append(res["best_asr"])
            data[attack_name]["acc"].append(res["best_acc"])
    return data

def compare_avg(our_list, baseline_list):
    our_avg = round(sum(our_list)/len(our_list),3)
    baseline_avg = round(sum(baseline_list)/len(baseline_list),3)
    '''
    if expect == "small":
        if our_avg < baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    else:
        if our_avg > baseline_avg:  # 满足期盼
            res = "Win"
        else:
            res = "Lose"
    '''
    return our_avg, baseline_avg

def main():
    p1_data = load_data(os.path.join(exp_root_dir,
                                            "CleanSeedWithPoison", "p_num=1", "results.json"))
    p2_data = load_data(os.path.join(exp_root_dir,
                                            "CleanSeedWithPoison", "p_num=2", "results.json"))
    
    trans_data = load_data(os.path.join(exp_root_dir,
                                            "CleanSeedWithPoison", "transSeed", "results.json"))
    
    for attack_name in attack_name_list:
        p1_PN_list = p1_data[attack_name]["pn"]
        p2_PN_list = p2_data[attack_name]["pn"]
        trans_PN_list = trans_data[attack_name]["pn"]
        p1_PN_avg, p2_PN_avg = compare_avg(p1_PN_list,p2_PN_list)
        trans_PN_avg = round(sum(trans_PN_list)/len(trans_PN_list),3)

        p1_asr_list = p1_data[attack_name]["asr"]
        p2_asr_list = p2_data[attack_name]["asr"]
        trans_asr_list = trans_data[attack_name]["asr"]
        p1_asr_avg, p2_asr_avg = compare_avg(p1_asr_list,p2_asr_list)
        trans_asr_avg = round(sum(trans_asr_list)/len(trans_asr_list),3)

        p1_acc_list = p1_data[attack_name]["acc"]
        p2_acc_list = p2_data[attack_name]["acc"]
        trans_acc_list = trans_data[attack_name]["acc"]
        p1_acc_avg, p2_acc_avg = compare_avg(p1_acc_list,p2_acc_list)
        trans_acc_avg = round(sum(trans_acc_list)/len(trans_acc_list),3)


        print(f"{attack_name}")
        print("\tPN")
        print(f"\t\tP1:{p1_PN_avg},P2:{p2_PN_avg},Trans:{trans_PN_avg}")

        print("\tASR")
        print(f"\t\tP1:{p1_asr_avg},P2:{p2_asr_avg},Trans:{trans_asr_avg}")

        print("\tACC")
        print(f"\t\tP1:{p1_acc_avg},P2:{p2_acc_avg},Trans:{trans_acc_avg}")




if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))
    main()