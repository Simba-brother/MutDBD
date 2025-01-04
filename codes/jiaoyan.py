'''
校验看看数据代码有没有错
'''
import os
import pandas as pd
import torch
from codes import config

def main():
    pass

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
    set_1 = set(poisoned_ids)
    pass_flag = True
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
        df_poisoned = preLabel_df.loc[preLabel_df["isPoisoned"]==True]
        set_2 = set(list(df_poisoned["sampled_id"]))
        if set_1 != set_2:
            print(f"rate:{rate} error")
            pass_flag = False
            break
    if pass_flag is True:
        print("pass")
    else:
        print("No pass")