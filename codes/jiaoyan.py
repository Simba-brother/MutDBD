'''
校验看看数据代码有没有错
'''
import os
import pandas as pd
import torch
from codes import config

def check(poisoned_ids,df):
    set_1 = set(poisoned_ids)
    df_poisoned = df.loc[df["isPoisoned"]==True]
    set_2 = set(list(df_poisoned["sampled_id"]))
    if set_1 != set_2:
        return "No pass"
    return "pass"

if __name__ == "__main__":
    # 加载后门数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    # 加载preLabel csv
    for rate in config.fine_mutation_rate_list:
        preLabel_csv_path = os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(rate),
                "preLabel.csv"
            )
        preLabel_df = pd.read_csv(preLabel_csv_path)
        check_flag = check(poisoned_ids,preLabel_df)
        if check_flag == "No pass":
            print(f"rate:{rate}没通过测试")