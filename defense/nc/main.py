import os
import time
import pprint
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR

from models.model_loader import get_model
from mid_data_loader import get_backdoor_data
from datasets.posisoned_dataset import get_all_dataset
from defense.our.sample_select import clean_seed
from utils.common_utils import convert_to_hms,set_random_seed
from utils.dataset_utils import get_class_num
from utils.common_utils import get_formattedDateTime
from utils.model_eval_utils import eval_asr_acc
from utils.save_utils import atomic_json_dump, load_results

def eval_and_save(model, filtered_poisoned_testset, clean_testset, device, save_path):
    asr, acc = eval_asr_acc(model,filtered_poisoned_testset,clean_testset,device)
    torch.save(model.state_dict(), save_path)
    return asr,acc


def freeze_model(model,dataset_name,model_name):
    if dataset_name == "CIFAR10" or dataset_name == "GTSRB":
        if model_name == "ResNet18":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'features.5' in name or 'features.4' in name or 'features.3' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name, param in model.named_parameters():
                if 'classifier' in name or 'linear' in name or 'dense4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    elif dataset_name == "ImageNet2012_subset":
        if model_name == "VGG19":
            for name, param in model.named_parameters():
                if 'classifier' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "DenseNet":
            for name,param in model.named_parameters():
                if 'classifier' in name or 'features.denseblock4' in name or 'features.denseblock3' in name:  # 只训练最后几层或全连接层
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif model_name == "ResNet18":
            for name,param in model.named_parameters():
                if 'fc' in name or 'layer4' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise Exception("模型不存在")
    else:
        raise Exception("模型不存在")
    return model
# ==================== Neural Cleanse 核心组件 ====================

class Normalize:
    """数据标准化"""
    def __init__(self, expected_values, variance, n_channels=3):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    """数据反标准化"""
    def __init__(self, expected_values, variance, n_channels=3):
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class RegressionModel(nn.Module):
    """触发器逆向工程模型"""
    def __init__(self, device, init_mask, init_pattern, classifier, normalizer=None, denormalizer=None, epsilon=1e-7):
        super(RegressionModel, self).__init__()
        self._EPSILON = epsilon
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask, dtype=torch.float32))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern, dtype=torch.float32))
        self.classifier = classifier
        self.normalizer = normalizer
        self.denormalizer = denormalizer

        # 冻结分类器参数
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(pattern)
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = torch.tanh(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = torch.tanh(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5


def get_backdoor_base_data(dataset_name, model_name, attack_name):
     # 加载后门攻击配套数据
    backdoor_data = get_backdoor_data(dataset_name, model_name, attack_name)
    if "backdoor_model" in backdoor_data.keys():
        backdoor_model = backdoor_data["backdoor_model"]
    else:
        model = get_model(dataset_name, model_name)
        state_dict = backdoor_data["backdoor_model_weights"]
        model.load_state_dict(state_dict)
        backdoor_model = model
    # 训练数据集中中毒样本id
    poisoned_ids = backdoor_data["poisoned_ids"]
    # filtered_poisoned_testset, poisoned testset中是所有的test set都被投毒了,为了测试真正的ASR，需要把poisoned testset中的attacked class样本给过滤掉
    poisoned_trainset, filtered_poisoned_testset, clean_trainset, clean_testset = get_all_dataset(dataset_name, model_name, attack_name, poisoned_ids)
    return backdoor_model,poisoned_ids, poisoned_trainset,filtered_poisoned_testset, clean_trainset, clean_testset


class Recorder:
    """记录优化过程中的最佳结果"""
    def __init__(self, init_cost=1e-3, cost_multiplier=2.0, patience=10,
                 early_stop=True, early_stop_threshold=1.0, early_stop_patience=5):
        super().__init__()
        # 最佳优化结果
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # 用于调整平衡成本的日志和计数器
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # 早停计数器
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # 成本参数
        self.cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.patience = patience
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.init_cost = init_cost

    def reset_state(self):
        self.cost = self.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print(f"Initialize cost to {self.cost:.6f}")


def train_step(regression_model, optimizer, dataloader, recorder, epoch, target_label,
               device, atk_succ_threshold=90.0, use_norm=1):
    """单个epoch的训练步骤"""
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    inner_early_stop_flag = False

    for batch_idx, (inputs, labels, isP) in enumerate(dataloader):
        optimizer.zero_grad()
        inputs = inputs.to(device).float()
        sample_num = inputs.shape[0]
        total_pred += sample_num

        target_labels = torch.ones((sample_num), dtype=torch.int64).to(device) * target_label
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), use_norm)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizer.step()

        minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)
        true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # 保存最佳mask
    if avg_loss_acc >= atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg

    # 早停检查
    if recorder.early_stop:
        if recorder.reg_best < float("inf"):
            if recorder.reg_best >= recorder.early_stop_threshold * recorder.early_stop_reg_best:
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0
        recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

        if (recorder.cost_down_flag and recorder.cost_up_flag and
            recorder.early_stop_counter >= recorder.early_stop_patience):
            print("Early stop!")
            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # 成本调整
        if recorder.cost == 0 and avg_loss_acc >= atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= recorder.patience:
                recorder.reset_state()
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= recorder.patience:
            recorder.cost_up_counter = 0
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True
        elif recorder.cost_down_counter >= recorder.patience:
            recorder.cost_down_counter = 0
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    del predictions
    torch.cuda.empty_cache()

    return inner_early_stop_flag


def train_mask(classifier, dataloader, target_label, device, dataset_name, attack_name,
               nc_epoch=10, mask_lr=0.1, init_cost=1e-3):
    """为特定目标标签训练触发器"""
    # 获取数据形状
    sample_batch = next(iter(dataloader))
    input_shape = sample_batch[0].shape[1:]  # (C, H, W)

    # 初始化mask和pattern
    init_mask = np.random.randn(1, *input_shape)
    init_pattern = np.random.randn(1, *input_shape)

    # 根据数据集和攻击类型获取normalizer和denormalizer
    if dataset_name.upper() == "CIFAR10":
        if attack_name == "IAD":
            normalizer = Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261], n_channels=3)
            denormalizer = Denormalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261], n_channels=3)
        elif attack_name in ["Refool", "LabelConsistent"]:
            normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
            denormalizer = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
        else:  # BadNets, WaNet
            normalizer = None
            denormalizer = None
    elif dataset_name.upper() == "GTSRB":
        if attack_name in ["Refool", "LabelConsistent"]:
            normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
            denormalizer = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
        else:  # BadNets, IAD, WaNet
            normalizer = None
            denormalizer = None
    elif dataset_name.upper() == "IMAGENET2012_SUBSET":
        if attack_name in ["Refool", "LabelConsistent"]:
            normalizer = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
            denormalizer = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], n_channels=3)
        else:  # BadNets, IAD, WaNet
            normalizer = None
            denormalizer = None
    else:
        normalizer = None
        denormalizer = None

    # 构建回归模型
    regression_model = RegressionModel(device, init_mask, init_pattern, classifier,
                                      normalizer, denormalizer).to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(regression_model.parameters(), lr=mask_lr, betas=(0.5, 0.9))

    # 设置记录器
    recorder = Recorder(init_cost=init_cost)

    for epoch in range(nc_epoch):
        early_stop = train_step(regression_model, optimizer, dataloader, recorder,
                               epoch, target_label, device)
        if early_stop:
            break

    return recorder.mask_best, recorder.pattern_best, recorder.reg_best


def outlier_detection(l1_norm_list, idx_mapping):
    """基于MAD的异常检测"""
    print("-" * 30)
    print("判断模型是否为后门模型")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print(f"Median: {median:.4f}, MAD: {mad:.4f}")
    print(f"Anomaly index: {min_mad:.4f}")

    if min_mad < 2:
        print("不是后门模型")
        is_backdoor = False
    else:
        print("这是一个后门模型")
        is_backdoor = True

    # 标记可疑标签
    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(f"可疑标签列表: {flag_list}")

    return is_backdoor, flag_list


def apply_trigger(img_tensor, mask, pattern):
    """将触发器应用到图像上"""
    # img_tensor: (C, H, W)
    # mask: (1, C, H, W)
    # pattern: (1, C, H, W)
    triggered_img = (1 - mask.squeeze(0)) * img_tensor + mask.squeeze(0) * pattern.squeeze(0)
    return triggered_img


def ours_train(model,device, dataset, num_epoch=30, lr=1e-3, batch_size=64,
          lr_scheduler=None, class_weight = None, weight_decay=None, early_stop=False):
    model.train()
    model.to(device)
    dataset_loader = DataLoader(
            dataset, # 非预制
            batch_size=batch_size,
            shuffle=True, # 打乱
            num_workers=4)
    if weight_decay:
        optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    if lr_scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
    elif lr_scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer,T_max=num_epoch,eta_min=1e-6)
    if class_weight is None:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss(class_weight.to(device))
    loss_function.to(device)
    optimal_loss = float('inf')
    best_model = model
    patience = 5
    count = 0
    for epoch in range(num_epoch):
        step_loss_list = []
        for _, batch in enumerate(dataset_loader):
            optimizer.zero_grad()
            X = batch[0].to(device)
            Y = batch[1].to(device)
            P_Y = model(X)
            loss = loss_function(P_Y, Y)
            loss.backward()
            optimizer.step()
            step_loss_list.append(loss.item())
        if lr_scheduler:
            scheduler.step()
        epoch_loss = sum(step_loss_list) / len(step_loss_list)
        print(f"epoch:{epoch},loss:{epoch_loss}")
        if epoch_loss < optimal_loss:
            count = 0
            optimal_loss = epoch_loss
            best_model = copy.deepcopy(model)
        else:
            count += 1
            if early_stop and count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return model,best_model


def one_scence(dataset_name, model_name, attack_name, save_dir=None):
    # 先得到后门攻击基础数据
    backdoor_model,poisoned_ids,poisoned_trainset,filtered_poisoned_testset, clean_trainset, clean_testset = \
        get_backdoor_base_data(dataset_name, model_name, attack_name)

    backdoor_model = backdoor_model.to(device)
    backdoor_model.eval()

    # 获取类别数
    num_classes = get_class_num(dataset_name)

    # 创建数据加载器（使用clean_trainset的一个子集进行触发器逆向）
    batch_size = 16
    clean_seedSet, _ = clean_seed(poisoned_trainset,poisoned_ids,strict_clean=True)
    nc_dataloader = DataLoader(clean_seedSet, batch_size=batch_size, shuffle=True, num_workers=4)

    # 为每个类别训练触发器
    print(f"\n为每个类别训练触发器 (共{num_classes}个类别)...")
    mask_list = []
    pattern_list = []
    l1_norm_list = []
    idx_mapping = {}

    nc_epoch = 10  # 每个类别的训练轮数
    mask_lr = 0.1  # mask学习率

    for target_label in range(num_classes):
        mask, pattern, reg = train_mask(
            backdoor_model, nc_dataloader, target_label, device,
            dataset_name, attack_name, nc_epoch=nc_epoch, mask_lr=mask_lr
        )

        mask_list.append(mask)
        pattern_list.append(pattern)
        l1_norm = torch.norm(mask, p=1)
        l1_norm_list.append(l1_norm)
        idx_mapping[target_label] = target_label

    # 转换为tensor
    l1_norm_list = torch.stack(l1_norm_list)

    # target class rank
    flag_list = []
    for target_label in range(num_classes):
        flag_list.append((target_label, l1_norm_list[target_label]))
    flag_list = sorted(flag_list, key=lambda x: x[1])
    
    class_rank_rate = 0
    class_rank_list = []
    for i,(target_label, l1_norm) in enumerate(flag_list):
        print(f"target_label:{target_label},l1_norm:{l1_norm}")
        class_rank_list.append(target_label)
        if target_label == gt_target_label:
            class_rank_rate = round((i+1) / num_classes, 4)
    print(f"class_rank_rate:{class_rank_rate*100}%")


    # # 异常检测
    # print("\n" + "="*60)
    # is_backdoor, flag_list = outlier_detection(l1_norm_list, idx_mapping)
    
    backdoor_target_label = flag_list[0][0]  # 最可疑的标签
    print(f"最可疑后门目标标签: {backdoor_target_label}")

    # ==================== 准备Unlearning数据集 ====================
    print("\n准备Unlearning数据集...")

    # 获取逆向出来的mask和pattern
    backdoor_mask = mask_list[backdoor_target_label].cpu()
    backdoor_pattern = pattern_list[backdoor_target_label].cpu()

    # 将数据集分为clean samples和unlearning samples
    unlearning_ratio = 0.2  # 20%用于unlearning
    total_samples = len(clean_seedSet)
    num_clean = int(total_samples * (1 - unlearning_ratio))
    num_unlearn = total_samples - num_clean

    print(f"Clean samples: {num_clean}, Unlearning samples: {num_unlearn}")

    # 创建新的数据集
    x_new = []
    y_new = []

    # 添加clean samples
    for i in range(num_clean):
        img, label, isP = clean_seedSet[i]
        x_new.append(img)
        y_new.append(label)

    # 添加unlearning samples（应用触发器但保持原始标签）
    for i in range(num_clean, total_samples):
        img, label, isP = clean_seedSet[i]
        # 应用触发器
        triggered_img = apply_trigger(img, backdoor_mask, backdoor_pattern)
        x_new.append(triggered_img)
        y_new.append(label) # 保持原始标签！

    # 创建新的数据集
    x_tensor = torch.stack(x_new)
    y_tensor = torch.tensor(y_new)
    finetune_dataset = TensorDataset(x_tensor, y_tensor)
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # ==================== 模型Unlearning ====================
    '''
    print("\n开始模型Unlearning（微调）...")
    for param in backdoor_model.parameters():
        param.requires_grad = True
    # freeze_model(backdoor_model,dataset_name,model_name)
    last_defense_model,best_defense_model = ours_train(backdoor_model,device,finetune_dataset,
               num_epoch=args["finetune"]["num_epoch"],
               lr=args["finetune"]["init_lr"],
               batch_size=args["finetune"]["batch_size"],
               early_stop=args["finetune"]["early_stop"]
                )
    defense_model = best_defense_model
    '''



    backdoor_model.train()
    optimizer = optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    finetune_epochs = 10
    for epoch in range(finetune_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in finetune_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs = backdoor_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{finetune_epochs}, Loss: {total_loss/len(finetune_loader):.4f}, TrainAcc: {train_acc:.2f}%")
    defense_model = backdoor_model
    print("模型微调完成")
    

    # ==================== 评估修复后的模型 ====================
    
    print("评估修复后的模型性能...")
    if save_model:
        save_path = os.path.join(save_dir, "defense_model.pth")
        asr,acc = eval_and_save(defense_model, filtered_poisoned_testset, clean_testset, device, save_path)
        print(f"防御模型权重保存在:{save_path}")
    else:
        asr, acc = eval_asr_acc(defense_model,filtered_poisoned_testset,clean_testset,device) 
    print(f"ACC: {acc:.3f}%")
    print(f"ASR: {asr:.3f}%")
    res = {
        "class_rank_list":class_rank_list,
        "class_rank_rate":class_rank_rate,
        "acc":acc,
        "asr":asr
    }
    return res


def save_experiment_result(exp_save_path, 
                           dataset_name, model_name, attack_name,r_seed,
                           result_data
                          ):
    """
    保存单个实验结果到嵌套JSON
    结构: {dataset: {model: {attack: {beta: {r_seed: result}}}}}
    """
    # 加载现有数据
    data = load_results(exp_save_path)

    # 构建嵌套结构
    if dataset_name not in data:
        data[dataset_name] = {}
    if model_name not in data[dataset_name]:
        data[dataset_name][model_name] = {}
    if attack_name not in data[dataset_name][model_name]:
        data[dataset_name][model_name][attack_name] = {}

    # 保存结果
    data[dataset_name][model_name][attack_name][str(r_seed)] = result_data

    # 原子写入
    atomic_json_dump(data, exp_save_path)

if __name__ == "__main__":
    # one-scence
    # exp_root_dir = "/data/mml/backdoor_detect/experiments"
    # dataset_name= "CIFAR10" # CIFAR10, GTSRB, ImageNet2012_subset
    # model_name= "ResNet18" # ResNet18, VGG19, DenseNet
    # attack_name ="BadNets" # BadNets, IAD, Refool, WaNet
    # gpu_id = 1
    # r_seed = 1

    # device = torch.device(f"cuda:{gpu_id}")
    # start_time = time.perf_counter()
    # print("="*60)
    # print(f"Neural Cleanse|{dataset_name}|{model_name}|{attack_name}|r_seed:{r_seed}")
    # set_random_seed(r_seed)
    # save_dir = os.path.join(exp_root_dir,"Defense","NC",dataset_name,model_name,attack_name)
    # os.makedirs(save_dir,exist_ok=True)
    # one_scence(dataset_name, model_name, attack_name, save_dir)
    # end_time = time.perf_counter()
    # cost_time = end_time - start_time
    # hours, minutes, seconds = convert_to_hms(cost_time)
    # print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")

    # all-scence
    cur_pid = os.getpid()
    exp_name = "NC"
    exp_time = get_formattedDateTime()
    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    exp_save_dir = os.path.join(exp_root_dir,"Defense", exp_name)
    exp_save_file_name = f"results_{exp_time}.json"
    exp_save_path = os.path.join(exp_save_dir,exp_save_file_name)
    save_model = False
    save_json = True

    exp_root_dir = "/data/mml/backdoor_detect/experiments"
    gt_target_label = 3
    dataset_name_list =  ["CIFAR10", "GTSRB","ImageNet2012_subset"] # [] # "ImageNet2012_subset"
    model_name_list =  ["ResNet18","VGG19","DenseNet"]
    attack_name_list = ["BadNets","IAD","Refool","WaNet"]
    r_seed_list = list(range(1,11))
    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}")

    args = {
        "finetune":{
            "method":"ours",
            "num_epoch":10,
            "init_lr":1e-4,
            "batch_size":64,
            "early_stop":False
        }
    }

    # 实验基础信息
    print("PID:",cur_pid)
    print("exp_root_dir:",exp_root_dir)
    print("exp_name:",exp_name)
    print("exp_time:",exp_time)
    print("exp_save_path:",exp_save_path)
    print("save_model:",save_model)
    print("save_json:",save_json)
    print("dataset_name_list:",dataset_name_list)
    print("model_name_list:",model_name_list)
    print("attack_name_list:",attack_name_list)
    print("r_seed_list:",r_seed_list)
    print("gpu_id:",gpu_id)

    # 实验超参数
    pprint.pprint(args)



    all_start_time = time.perf_counter()
    for r_seed in r_seed_list:
        set_random_seed(r_seed)
        one_repeat_start_time = time.perf_counter()
        for dataset_name in dataset_name_list:
            for model_name in model_name_list:
                for attack_name in attack_name_list:
                    if dataset_name == "ImageNet2012_subset" and model_name == "VGG19":
                        continue
                    one_sence_start_time = time.perf_counter()
                    print(f"\n{dataset_name}|{model_name}|{attack_name}|r_seed={r_seed}|timestamp:{get_formattedDateTime()}")
                    if save_model:
                        save_dir = os.path.join(exp_save_dir,dataset_name,model_name,attack_name,f"exp_{r_seed}")
                        os.makedirs(save_dir,exist_ok=True)
                    else:
                        save_dir = None
                    res = one_scence(dataset_name, model_name, attack_name, save_dir)
                    if save_json:
                        save_experiment_result(exp_save_path, 
                           dataset_name, model_name, attack_name,r_seed,
                           res)
                    one_scence_end_time = time.perf_counter()
                    one_scence_cost_time = one_scence_end_time - one_sence_start_time
                    hours, minutes, seconds = convert_to_hms(one_scence_cost_time)
                    print(f"one-scence耗时:{hours}时{minutes}分{seconds:.1f}秒")
        one_repeat_end_time = time.perf_counter()
        one_repeart_cost_time = one_repeat_end_time - one_repeat_start_time
        hours, minutes, seconds = convert_to_hms(one_repeart_cost_time)
        print(f"\n一轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
    all_end_time = time.perf_counter()
    all_cost_time = all_end_time - all_start_time
    hours, minutes, seconds = convert_to_hms(all_cost_time)
    print(f"\n{len(r_seed_list)}轮次全场景耗时:{hours}时{minutes}分{seconds:.1f}秒")
