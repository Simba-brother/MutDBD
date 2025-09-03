'''
评估模型性能
'''
import os
from commonUtils import read_yaml
from attack.models import get_model
from datasets.clean_dataset import get_clean_dataset
from modelEvalUtils import EvalModel
import torch

config = read_yaml("config.yaml")
exp_root_dir = config["exp_root_dir"]
dataset_name = "CIFAR10"
model_name = "ResNet18"
attack_name = "LabelConsistent"

benign_dict = {
    "ResNet18":"benign_train_2025-07-16_13:17:28",
    "VGG19":"benign_train_2025-07-16_17:35:57",
    "DenseNet":"benign_train_2025-07-16_22:34:07"
}
benign_state_dict_path = os.path.join(exp_root_dir,"ATTACK",dataset_name, model_name, attack_name, benign_dict[model_name], "best_model.pth")

model = get_model(dataset_name, model_name)
model.load_state_dict(benign_state_dict_path)

clean_trainset, clean_testset = get_clean_dataset(dataset_name,attack_name)
device = torch.device("cuda:0")
em = EvalModel(model, clean_testset, device)
acc = em.eval_acc()
print(acc)


