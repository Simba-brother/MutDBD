import os
import torch
from datasets.clean_dataset import get_clean_dataset
from models.model_loader import get_model
from mid_data_loader import get_labelConsistent_benign_model
from pgd import PGD
from torch.utils.data import DataLoader
from torch import nn
from modelEvalUtils import EvalModel,eval_asr_acc
from tqdm import tqdm
import random
from custom_dataset import CustomImageDataset
from torch.utils.data import Subset, ConcatDataset
from poisoning import PoisonedDataset
from utils.trainer import NeuralNetworkTrainer
from adv import construt_fusion_dataset
from datasets.utils import check_labels

# 读取原始数据集
dataset_name = "GTSRB"
attack_name = "LabelConsistent"
model_name = "ResNet18"
# 已经经过了transforms了
clean_trainset, clean_testset = get_clean_dataset(dataset_name,attack_name)

# 加载benign model
benign_state_dict = get_labelConsistent_benign_model(dataset_name,model_name)
victim_model = get_model(dataset_name, model_name)
victim_model.load_state_dict(benign_state_dict)
device = torch.device("cuda:0")
victim_model.to(device)

# 评估一下对抗前模型性能
em = EvalModel(victim_model,clean_trainset,device,batch_size=512, num_workers=8)
acc_clean = em.eval_acc()
print(f"模型在clean_trainset上的acc:{acc_clean}")

# 开始对抗攻击
target_class = 3
percent = 0.8
fusion_dataset,selected_indices,adv_indices = construt_fusion_dataset(victim_model,clean_trainset,device,target_class,percent)
# 包装投毒数据集
poisoned_trainset = PoisonedDataset(fusion_dataset,adv_indices, target_class, "train")
poisoned_testset = PoisonedDataset(clean_testset,list(range(len(clean_testset))), target_class, "test")

check_labels(poisoned_trainset)
check_labels(poisoned_testset)
# 创建训练器
trainer = NeuralNetworkTrainer(
    model=victim_model,
    device = device,
    init_lr = 0.01,
    experiment_name="gtsrb_resnet18"
)

poisoned_trainset_loader = DataLoader(
    poisoned_trainset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    drop_last=False,
    pin_memory=True
)


poisoned_testset_loader = DataLoader(
    poisoned_testset,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    drop_last=False,
    pin_memory=True
)

# 训练模型
history = trainer.fit(
    train_loader=poisoned_trainset_loader,
    val_loader=poisoned_testset_loader,  # 这里用测试集作为验证集，实际应用中应使用单独的验证集
    epochs=50,
    early_stopping_patience=3,
    save_best=True
)

asr, acc = eval_asr_acc(trainer.model,poisoned_testset,clean_testset,device)
print(f"ASR:{asr},ACC:{acc}")
exp_root_dir = "/data/mml/backdoor_detect/experiments"
backdoor_data = {}
backdoor_data["backdoor_model"] = trainer.model
backdoor_data["poisoned_ids"]=selected_indices
save_path = os.path.join(exp_root_dir, "ATTACK", dataset_name, model_name, attack_name,"backdoor_data.pth")
torch.save(backdoor_data, save_path)
print(f"LabelConsistent攻击完成,数据被存入{save_path}")














