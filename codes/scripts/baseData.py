import os
import joblib
exp_root_dir = "/data/mml/backdoor_detect/experiments"
class BaseData(object):
    def __init__(self, dataset_name, model_name, attack_name, mutation_operator_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.attack_name = attack_name
        self.mutation_operator_name = mutation_operator_name

    def _get_mutation_weight_file(self, mutation_rate):
        mutation_weight_file_list = []
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        if mutation_operator_name  == "gf":
            ratio_dir = f"ratio_{mutation_rate}_scale_5_num_50"
        else:
            ratio_dir = f"ratio_{mutation_rate}_num_50"
        mutated_models_weight_dir_path = os.path.join(exp_root_dir, dataset_name, model_name, "mutates", mutation_operator_name, ratio_dir, attack_name)
        for m_i in range(50):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"model_mutated_{m_i+1}.pth")
            mutation_weight_file_list.append(weight_path)
        return mutation_weight_file_list
    
    def _get_mutation_weight_file_2(self, mutation_rate):
        mutation_weight_file_list = []
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        mutated_models_weight_dir_path = os.path.join(exp_root_dir, "mutations", dataset_name, model_name, attack_name, mutation_operator_name, str(mutation_rate))
        for m_i in range(50):
            weight_path = os.path.join(mutated_models_weight_dir_path, f"mutated_model_{m_i}.pth")
            mutation_weight_file_list.append(weight_path)
        return mutation_weight_file_list
    
    def get_eval_poisoned_trainset_report_data(self):
        file_name = "eval_poisoned_trainset_report.data"
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        data_path = os.path.join(exp_root_dir,dataset_name, model_name, attack_name, mutation_operator_name, file_name)
        report = joblib.load(data_path)
        return report
    
    def get_eval_poisoned_trainset_target_class_report_data(self):
        file_name = "eval_poisoned_trainset_target_class.data"
        dataset_name = self.dataset_name
        model_name = self.model_name
        mutation_operator_name = self.mutation_operator_name
        attack_name = self.attack_name
        data_path = os.path.join(exp_root_dir,dataset_name, model_name, attack_name, mutation_operator_name, file_name)
        report = joblib.load(data_path)
        return report
    