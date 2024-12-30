import os
from codes import config

dataset_name_list = ["GTSRB"]
model_name_list = ["ResNet18","VGG19","DenseNet"]
attack_name_list = ["BadNets","IAD","Refool","WaNet"]
for dataset_name in dataset_name_list:
    for model_name in model_name_list:
        for attack_name in attack_name_list:
            for rate in config.fine_mutation_rate_list:
                dir = os.path.join(
                    config.exp_root_dir,
                    "EvalMutationToCSV",
                    dataset_name,
                    model_name,
                    attack_name,
                    str(rate))
                entries = os.listdir(dir)
                # 过滤出文件
                files = [entry for entry in entries if os.path.isfile(os.path.join(dir, entry))]
                file_name = files[0]
                old_file_path = os.path.join(dir,file_name)
                new_file_path = os.path.join(dir, "preLabel.csv")
                os.rename(old_file_path,new_file_path)
print("end")
    