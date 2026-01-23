import json
from datetime import datetime


def calcu_time_interval(start_time:str,end_time:str,fmt: str = "%Y-%m-%d %H:%M:%S")->str:
    
    start_dt = datetime.strptime(start_time.strip(), fmt)
    end_dt = datetime.strptime(end_time.strip(), fmt)
    delta_seconds = int((end_dt - start_dt).total_seconds())
    if delta_seconds < 0:
        raise ValueError("end_time must be >= start_time")

    hours = delta_seconds // 3600
    minutes = (delta_seconds % 3600) // 60
    seconds = delta_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    # 读取 cost_time.json
    with open(cost_time_json_path,mode="r") as f:
        cost_time_json = json.load(f)
    cost_time_ours = cost_time_json["ours"]
    for dataset_name in cost_time_ours.keys():
        for model_name in cost_time_ours[dataset_name].keys():
            for attack_name in cost_time_ours[dataset_name][model_name].keys():
                scence_time_dict = cost_time_ours[dataset_name][model_name][attack_name]
                start_time,end_time = scence_time_dict["class_rank"].split("to")
                cost_time = calcu_time_interval(start_time,end_time)
                print(f"{dataset_name}|{model_name}|{attack_name}: {cost_time}")

if __name__ == "__main__":
    cost_time_json_path = "cost_time.json"
    main()