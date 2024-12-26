import os
import joblib
from codes import config

if __name__ == "__main__":

    # 加载数据
    clean_report = joblib.load(os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "clean_report.data"
    ))

    poisoned_report = joblib.load(os.path.join(
        config.exp_root_dir, 
        "TargetClassAnalyse",
        config.dataset_name, 
        config.model_name, 
        config.attack_name,
        "poisoned_report.data"
    ))

    print("fa")
