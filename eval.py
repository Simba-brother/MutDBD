'''
评估模型性能
'''
from commonUtils import read_yaml
from attack.models import get_model
from datasets.clean_dataset import get_clean_dataset
from modelEvalUtils import EvalModel
import torch
from mid_data_loader import get_labelConsistent_benign_model, get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from modelEvalUtils import eval_asr_acc


def eval_LabelConsistent_benign_model(dataset_name, model_name):
    model = get_model(dataset_name, model_name)
    benign_state_dict = get_labelConsistent_benign_model(dataset_name, model_name)
    model.load_state_dict(benign_state_dict)

    clean_trainset, clean_testset = get_clean_dataset(dataset_name,"LabelConsistent")
    em = EvalModel(model, clean_testset, device)
    acc = em.eval_acc()
    return acc


def eval_Backdoor_model(dataset_name, model_name, attack_name):
    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    ASR, ACC = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_test_dataset,device)
    return ASR,ACC

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_root_dir = config["exp_root_dir"]
    device = torch.device("cuda:0")
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "LabelConsistent"
    ASR,ACC = eval_Backdoor_model(dataset_name, model_name, attack_name)
    print(f"{dataset_name}|{model_name}|{attack_name}")
    print(f"ASR:{ASR},ACC:{ACC}")
    pass

