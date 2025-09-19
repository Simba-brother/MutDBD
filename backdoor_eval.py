from attack.models import get_model
from mid_data_loader import get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from modelEvalUtils import EvalModel,eval_asr_acc
import torch
from datasets.utils import check_labels
dataset_name = "ImageNet2012_subset"
model_name = "VGG19"
attack_name = "WaNet"

# model = get_model(dataset_name,model_name)
# model.load_state_dict(torch.load("models/gtsrb_resnet18/best_model_epoch_6.pth",map_location="cpu")["model_state_dict"])
backdoor_data = get_backdoor_data(dataset_name,model_name,attack_name)
backdoor_model = backdoor_data["backdoor_model"]
poisoned_ids = backdoor_data["poisoned_ids"]
poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = \
    get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)

check_labels(filtered_poisoned_testset)
check_labels(clean_testset)


device = torch.device("cuda:0")
asr,acc = eval_asr_acc(backdoor_model,filtered_poisoned_testset,clean_testset,device)
print(f"backdoor model: asr:{asr},acc:{acc}")




