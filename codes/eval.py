import os

from codes import config
from codes.common.eval_model import EvalModel


# 加载后门攻击配套数据
backdoor_data_path = os.path.join(config.exp_root_dir, 
                                        "ATTACK", 
                                        config.dataset_name, 
                                        config.model_name, 
                                        config.attack_name, 
                                        "backdoor_data.pth")
backdoor_data = torch.load(backdoor_data_path,map_location="cpu")
backdoor_model = backdoor_data["backdoor_model"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_testset = backdoor_data["poisoned_testset"] # 预制的poisoned_testset