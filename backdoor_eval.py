import torch
from models.model_loader import get_model
from mid_data_loader import get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from utils.model_eval_utils import eval_asr_acc
from datasets.utils import check_labels


def main_one_sence(dataset_name,model_name,attack_name):
    # model = get_model(dataset_name,model_name)
    # model.load_state_dict(torch.load("models/gtsrb_resnet18/best_model_epoch_6.pth",map_location="cpu")["model_state_dict"])
    backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = \
        get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)

    filtered_poisoned_testset_label_count = check_labels(filtered_poisoned_testset)
    clean_testset_label_count = check_labels(clean_testset)
    device = torch.device("cuda:0")
    asr,acc = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_testset,device)
    return asr,acc,filtered_poisoned_testset_label_count,clean_testset_label_count

if __name__ == "__main__":
    
    '''
    dataset_name = "GTSRB"
    model_name = "DenseNet"
    attack_name = "IAD"
    asr,acc,filtered_poisoned_testset_label_count,clean_testset_label_count = main_one_sence(dataset_name, model_name, attack_name)
    print(f"sence:{dataset_name}|{model_name}|{attack_name}|ASR:{asr}|ACC:{acc}")
    print(f"ASR:{asr}|ACC:{acc}")
    print("filtered_poisoned_testset_label_count:")
    print(f"\t{filtered_poisoned_testset_label_count}")
    print("clean_testset_label_count:")
    print(f"\t{clean_testset_label_count}")
    print("END")
    '''

    dataset_name_list = ["CIFAR10", "GTSRB", "ImageNet2012_subset"]
    model_name_list = ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            if dataset_name_list == "ImageNet2012_subset" and model_name  == "VGG19":
                    continue
            for attack_name in attack_name_list:
                asr,acc,filtered_poisoned_testset_label_count,clean_testset_label_count = main_one_sence(dataset_name, model_name, attack_name)
                print(f"Sence:{dataset_name}|{model_name}|{attack_name}|ASR:{asr}|ACC:{acc}")
                print(f"ASR:{asr}|ACC:{acc}")
                print("filtered_poisoned_testset_label_count:")
                print(f"\t{filtered_poisoned_testset_label_count}")
                print("clean_testset_label_count:")
                print(f"\t{clean_testset_label_count}")
    
    






