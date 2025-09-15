import torch
from datasets.clean_dataset import get_clean_dataset
from attack.models import get_model
from mid_data_loader import get_labelConsistent_benign_model
from pgd import PGD
from torch.utils.data import DataLoader
from torch import nn
from modelEvalUtils import EvalModel
from tqdm import tqdm
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
print(f"acc_clean:{acc_clean}")

# 数据加载器
batch_size = 128
data_loader = DataLoader(
    clean_trainset,
    batch_size=batch_size,
    shuffle=False, # importent 
    num_workers=4,
    drop_last=False,
    pin_memory=True
)
loss_fn = nn.CrossEntropyLoss()
steps = 100
alpha = 0.01
epsilon = 0.3 # 0.3
# 开始对抗攻击
pgd = PGD(victim_model,loss_fn,steps,alpha,epsilon)
success_num = 0 
total = 0
all_adv_X = []
all_labels = []
all_pred_labels = []
for batch_id, batch in enumerate(tqdm(data_loader,desc="Processing batches")):
    X = batch[0].to(device)
    Y = batch[1].to(device)
    total += X.shape[0]
    adv_X = pgd.perturb(X,Y,device)
    with torch.no_grad():
        outputs = victim_model(adv_X)
        pred_Y = torch.argmax(outputs, dim=1)
        all_adv_X.append(adv_X)
        all_labels.append(Y)
        all_pred_labels.append(pred_Y)
        success_num += (pred_Y != Y).sum()
combined_adv_X = torch.cat(all_adv_X,dim=0)
combined_labels = torch.cat(all_labels,dim=0)
combined_pred_labels = torch.cat(all_pred_labels,dim=0)

mapping_list = (combined_labels == combined_pred_labels).tolist()
# 获取所有 True 值的索引
unsuccess_indices = set([i for i, value in enumerate(mapping_list) if value is True])
asr = round(success_num.item()/total,4)
print(f"asr:{asr}")
print(f"unsuccess_indices_nums:{len(unsuccess_indices)}")





