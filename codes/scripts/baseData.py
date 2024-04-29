'''
加载一些基本的保存数据脚本
'''

import os
import joblib
# 实验根目录
exp_root_dir = "/data/mml/backdoor_detect/experiments"

# 基本数据加载器
class BaseData(object):
    def __init__(self, dataset_name, model_name, attack_name, mutation_operator_name):
        # 数据集名称
        self.dataset_name = dataset_name 
        # 模型名称
        self.model_name = model_name
        # 攻击名称
        self.attack_name = attack_name
        # 变异算子名称
        self.mutation_operator_name = mutation_operator_name
        # 变异模型数量
        self.mutation_model_num = 50

    # def _get_mutation_weight_file(self, mutation_rate):
    #     mutation_weight_file_list = []
    #     dataset_name = self.dataset_name
    #     model_name = self.model_name
    #     mutation_operator_name = self.mutation_operator_name
    #     attack_name = self.attack_name
    #     if mutation_operator_name  == "gf":
    #         ratio_dir = f"ratio_{mutation_rate}_scale_5_num_50"
    #     else:
    #         ratio_dir = f"ratio_{mutation_rate}_num_50"
    #     mutated_models_weight_dir_path = os.path.join(exp_root_dir, dataset_name, model_name, "mutates", mutation_operator_name, ratio_dir, attack_name)
    #     for m_i in range(50):
    #         weight_path = os.path.join(mutated_models_weight_dir_path, f"model_mutated_{m_i+1}.pth")
    #         mutation_weight_file_list.append(weight_path)
    #     return mutation_weight_file_list
    
    def get_mutation_weight_file_by_mutation_rate(self, mutation_rate):
        '''
        获得某个变异算子下特定变异率的权重list。
        '''
        mutation_weight_file_list = []
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        mutated_models_weight_dir_path = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
        for m_i in range(self.mutation_model_num):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"mutated_model_{m_i}.pth")
            mutation_weight_file_list.append(weight_path)
        return mutation_weight_file_list
    
    def get_eval_poisoned_trainset_report_data(self):
        # 获得变异算子下变异模型在poisoned_trainset上的评估report
        file_name = "eval_poisoned_trainset_report.data"
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        data_path = os.path.join(
            exp_root_dir,
            dataset_name, 
            model_name, 
            attack_name, 
            mutation_operator_name, 
            file_name)
        report = joblib.load(data_path)
        return report
    
    def get_eval_poisoned_trainset_target_class_report_data(self):
        # 获得变异算子下变异模型在target class of poisoned_trainset上的评估report
        file_name = "eval_poisoned_trainset_target_class.data"
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        data_path = os.path.join(
            exp_root_dir,
            dataset_name, 
            model_name, 
            attack_name, 
            mutation_operator_name, 
            file_name)
        report = joblib.load(data_path)
        return report
    

if __name__ == "__main__":
    pass