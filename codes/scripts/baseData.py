import os
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