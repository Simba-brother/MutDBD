from defense.our.sample_select import chose_retrain_set
from mid_data_loader import get_backdoor_data,get_our_method_state,get_asd_method_state,get_class_rank
from datasets.posisoned_dataset import get_all_dataset
from utils.common_utils import read_yaml
import torch
from models.model_loader import get_model
from utils.model_eval_utils import eval_asr_acc


def get_defense_and_select_model(defensed_state_dict_path, selected_state_dict_path, dataset_name, model_name):
    defense_model = get_model(dataset_name, model_name)
    select_model = get_model(dataset_name, model_name)
    defense_model.load_state_dict(torch.load(defensed_state_dict_path, map_location="cpu"))
    select_model.load_state_dict(torch.load(selected_state_dict_path, map_location="cpu"))
    return defense_model, select_model

def get_pn_asr_acc(dataset_name, model_name, attack_name, method_name, poisoned_testset, clean_testset, poisoned_trainset, poisoned_ids, r_seed):
    if method_name == "Our":
        class_rank = get_class_rank(dataset_name,model_name,attack_name)
        defensed_state_dict_path, selected_state_dict_path = get_our_method_state(dataset_name, model_name, attack_name, r_seed)
        defense_model, select_model = get_defense_and_select_model(defensed_state_dict_path, selected_state_dict_path)
        ASR,ACC = eval_asr_acc(defense_model,poisoned_testset,clean_testset,device)
        choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = chose_retrain_set(select_model, device, 
                        choice_rate, poisoned_trainset, poisoned_ids, class_rank=class_rank)
        return PN,ASR,ACC
    elif method_name == "ASD":
        defensed_state_dict_path, selected_state_dict_path = get_asd_method_state(dataset_name, model_name, attack_name, r_seed)
        defense_model, select_model = get_defense_and_select_model(defensed_state_dict_path, selected_state_dict_path)
        ASR,ACC = eval_asr_acc(defense_model,poisoned_testset,clean_testset,device)
        choicedSet,choiced_sample_id_list,remainSet,remain_sample_id_list, PN = chose_retrain_set(select_model, device, 
                        choice_rate, poisoned_trainset, poisoned_ids, class_rank=None)
        return PN,ASR,ACC

def one_science(dataset_name,model_name,attack_name):
    
    backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids = backdoor_data["poisoned_ids"]

    poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset = get_all_dataset(dataset_name,model_name,attack_name,poisoned_ids)
    # 获得ranker_model
    for r_seed in range(1,11):
        print(f"r_seed:{r_seed}")
        # OurMethod
        PN,ASR,ACC = get_pn_asr_acc(dataset_name, model_name, attack_name, "Our", filtered_poisoned_testset, clean_test_dataset, poisoned_trainset, poisoned_ids, r_seed)
        print(f"Our|PN:{PN},ASR:{ASR},ACC:{ACC}")
        PN,ASR,ACC = get_pn_asr_acc(dataset_name, model_name, attack_name, "ASD", filtered_poisoned_testset, clean_test_dataset, poisoned_trainset, poisoned_ids, r_seed)
        print(f"ASD|PN:{PN},ASR:{ASR},ACC:{ACC}")

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_root_dir = config["exp_root_dir"]
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "BadNets"
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")
    choice_rate = 0.6
    print(f"{dataset_name}|{model_name}|{attack_name}")
    print(f"choice_rate:{choice_rate}")
    one_science(dataset_name,model_name,attack_name)
