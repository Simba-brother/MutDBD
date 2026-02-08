
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
    cut_train_data = load_data(os.path.join(exp_root_dir,
                                            "small_dataset_defense_train", "results.json"))
    semi_train_data = load_data(os.path.join(exp_root_dir,
                                             "small_dataset_defense_semitrain", "results.json"))

    for attack_name in attack_name_list:
        cut_asr_list = cut_train_data[attack_name]["asr"]
        semi_asr_list = semi_train_data[attack_name]["asr"]

        cut_asr_avg, semi_asr_avg = compare_avg(cut_asr_list,semi_asr_list)

        cut_acc_list = cut_train_data[attack_name]["acc"]
        semi_acc_list = semi_train_data[attack_name]["acc"]

        cut_acc_avg, semi_acc_avg = compare_avg(cut_acc_list,semi_acc_list)


        print(f"{attack_name}")
        print("\tASR")
        print(f"\t\tsupervised:{cut_asr_avg},semi:{semi_asr_avg}")

        print("\tACC")
        print(f"\t\tsupervised:{cut_acc_avg},semi:{semi_acc_avg}")





if __name__ == "__main__":
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))
    main()