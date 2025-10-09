from utils.common_utils import read_yaml
from models.model_loader import get_model
from datasets.clean_dataset import get_clean_dataset
from utils.model_eval_utils import EvalModel
import torch
from mid_data_loader import get_labelConsistent_benign_model, get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from utils.model_eval_utils import eval_asr_acc


def eval_LabelConsistent_benign_model(dataset_name, model_name):
    benign_model = get_model(dataset_name, model_name)
    benign_state_dict = get_labelConsistent_benign_model(dataset_name, model_name)
    benign_model.load_state_dict(benign_state_dict)

    trainset, testset = get_clean_dataset(dataset_name,attack_name)
    em = EvalModel(benign_model, testset, device)
    acc = em.eval_acc()
    return acc


def eval_Backdoor_and_Benign_model_For_LabelConsistent(dataset_name, model_name):
    backdoor_data = get_backdoor_data(dataset_name, model_name, "LabelConsistent")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_trainset, filtered_poisoned_testset, clean_train_dataset, clean_test_dataset = get_all_dataset(dataset_name, model_name, "LabelConsistent", poisoned_ids)
    bd_ASR, bd_ACC = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_test_dataset,device)

    benign_model = get_model(dataset_name, model_name)
    benign_state_dict = get_labelConsistent_benign_model(dataset_name, model_name)
    benign_model.load_state_dict(benign_state_dict)
    be_ASR, be_ACC = eval_asr_acc(benign_model,filtered_poisoned_testset,clean_test_dataset,device)

    return bd_ASR, bd_ACC, be_ASR, be_ACC

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_root_dir = config["exp_root_dir"]
    device = torch.device("cuda:0")
    dataset_name = "GTSRB"
    model_name = "ResNet18"
    attack_name = "LabelConsistent"
    acc = eval_LabelConsistent_benign_model(dataset_name, model_name)
    print(f"be_ACC:{acc}")
    # bd_ASR, bd_ACC, be_ASR, be_ACC = eval_Backdoor_and_Benign_model_For_LabelConsistent(dataset_name, model_name)
    # print(f"{dataset_name}|{model_name}|{attack_name}")
    # print(f"bd_ASR:{bd_ASR},bd_ACC:{bd_ACC},be_ASR:{be_ASR},be_ACC:{be_ACC}")
    pass

