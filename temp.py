
import os
import shutil
from codes import config


def main():
    source_file_path = os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV_confidence",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        "0.05",
        "confidence.csv"
    )

    target_dir_path = os.path.join(
        config.exp_root_dir,
        "EvalMutationToCSV",
        config.dataset_name,
        config.model_name,
        config.attack_name,
        "0.05"
    )

    shutil.move(source_file_path,target_dir_path)
    print(f"target_dir_path:{target_dir_path}")

    

if __name__ == "__main__":
    main()