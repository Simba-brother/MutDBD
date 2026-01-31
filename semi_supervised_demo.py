"""
半监督学习Demo - 理解监督+无监督训练原理

核心思想：
1. 有标签数据：用真实标签进行监督学习
2. 无标签数据：用模型预测的"伪标签"进行学习
3. 通过一致性正则化：让模型对同一样本的不同增强版本给出一致的预测

本demo实现了简化版的MixMatch算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random


# ==================== 1. 数据集准备 ====================

class SemiSupervisedDataset(Dataset):
    """
    半监督数据集包装器

    Args:
        dataset: 原始数据集
        labeled_indices: 有标签样本的索引列表
        is_labeled: True表示返回有标签数据，False表示返回无标签数据
    """
    def __init__(self, dataset, labeled_indices, is_labeled=True):
        self.dataset = dataset
        self.is_labeled = is_labeled

        # 创建标记数组：1表示有标签，0表示无标签
        self.label_mask = np.zeros(len(dataset), dtype=int)
        self.label_mask[labeled_indices] = 1

        if is_labeled:
            # 有标签数据的索引
            self.indices = np.where(self.label_mask == 1)[0]
        else:
            # 无标签数据的索引
            self.indices = np.where(self.label_mask == 0)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]

        if self.is_labeled:
            # 有标签数据：返回图像和真实标签
            return {
                'image': img,
                'label': label,
                'is_labeled': True
            }
        else:
            # 无标签数据：返回两个不同增强的图像（用于一致性正则化）
            # 注意：这里简化了，实际应该用不同的数据增强
            img2, _ = self.dataset[real_idx]
            return {
                'image1': img,
                'image2': img2,
                'label': label,  # 仅用于评估，训练时不使用
                'is_labeled': False
            }


# ==================== 2. 简单的CNN模型 ====================

class SimpleCNN(nn.Module):
    """简单的CNN分类器"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 3. 半监督训练核心函数 ====================

def sharpen(predictions, temperature=0.5):
    """
    锐化预测分布（Temperature Sharpening）

    原理：降低预测的熵，让模型更有信心
    - temperature < 1: 让高概率更高，低概率更低
    - temperature = 1: 不变
    - temperature > 1: 让分布更平滑

    例如：[0.7, 0.2, 0.1] -> [0.85, 0.10, 0.05]
    """
    predictions = predictions ** (1 / temperature)
    return predictions / predictions.sum(dim=1, keepdim=True)


def mixup(x1, x2, y1, y2, alpha=0.75):
    """
    MixUp数据增强

    原理：线性插值混合两个样本
    mixed_x = λ * x1 + (1-λ) * x2
    mixed_y = λ * y1 + (1-λ) * y2

    作用：增强模型的泛化能力，让决策边界更平滑
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # 确保 lam >= 0.5

    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2

    return mixed_x, mixed_y


def semi_supervised_train_epoch(model, labeled_loader, unlabeled_loader,
                                optimizer, device, epoch, num_classes=10,
                                lambda_u=10.0):
    """
    半监督训练一个epoch

    Args:
        model: 模型
        labeled_loader: 有标签数据加载器
        unlabeled_loader: 无标签数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        num_classes: 类别数
        lambda_u: 无标签损失的权重

    返回:
        平均损失
    """
    model.train()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    total_loss = 0
    total_labeled_loss = 0
    total_unlabeled_loss = 0
    num_batches = 0

    # 迭代次数取决于较小的数据加载器
    num_iters = min(len(labeled_loader), len(unlabeled_loader))

    for batch_idx in range(num_iters):
        try:
            labeled_batch = next(labeled_iter)
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            break

        # ========== 步骤1: 准备有标签数据 ==========
        x_labeled = labeled_batch['image'].to(device)
        y_labeled = labeled_batch['label'].to(device)
        batch_size = x_labeled.size(0)

        # 转换为one-hot编码
        y_labeled_onehot = torch.zeros(batch_size, num_classes).to(device)
        y_labeled_onehot.scatter_(1, y_labeled.view(-1, 1), 1)

        # ========== 步骤2: 为无标签数据生成伪标签 ==========
        x_unlabeled1 = unlabeled_batch['image1'].to(device)
        x_unlabeled2 = unlabeled_batch['image2'].to(device)

        with torch.no_grad():
            # 对同一样本的两个增强版本进行预测
            pred1 = model(x_unlabeled1)
            pred2 = model(x_unlabeled2)

            # 平均两个预测（集成效果）
            pred_avg = (F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1)) / 2

            # 锐化预测分布，生成伪标签
            pseudo_labels = sharpen(pred_avg, temperature=0.5)

        # ========== 步骤3: MixUp数据增强 ==========
        # 将有标签和无标签数据混合
        all_images = torch.cat([x_labeled, x_unlabeled1, x_unlabeled2], dim=0)
        all_labels = torch.cat([y_labeled_onehot, pseudo_labels, pseudo_labels], dim=0)

        # 随机打乱
        indices = torch.randperm(all_images.size(0))
        mixed_images, mixed_labels = mixup(
            all_images, all_images[indices],
            all_labels, all_labels[indices],
            alpha=0.75
        )

        # ========== 步骤4: 前向传播和损失计算 ==========
        # 分离有标签和无标签部分
        logits = model(mixed_images)
        logits_labeled = logits[:batch_size]
        logits_unlabeled = logits[batch_size:]

        # 有标签损失：交叉熵
        labeled_loss = -torch.mean(
            torch.sum(F.log_softmax(logits_labeled, dim=1) * mixed_labels[:batch_size], dim=1)
        )

        # 无标签损失：MSE（让预测接近伪标签）
        unlabeled_probs = F.softmax(logits_unlabeled, dim=1)
        unlabeled_loss = torch.mean(
            (unlabeled_probs - mixed_labels[batch_size:]) ** 2
        )

        # 总损失 = 有标签损失 + λ * 无标签损失
        # λ控制无标签数据的影响程度
        loss = labeled_loss + lambda_u * unlabeled_loss

        # ========== 步骤5: 反向传播和优化 ==========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_labeled_loss += labeled_loss.item()
        total_unlabeled_loss += unlabeled_loss.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{num_iters} | "
                  f"Loss: {loss.item():.4f} | "
                  f"L_x: {labeled_loss.item():.4f} | "
                  f"L_u: {unlabeled_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_labeled_loss = total_labeled_loss / num_batches
    avg_unlabeled_loss = total_unlabeled_loss / num_batches

    return avg_loss, avg_labeled_loss, avg_unlabeled_loss


def supervised_train_epoch(model, train_loader, optimizer, device):
    """
    纯监督训练一个epoch（用于对比）
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    return total_loss / num_batches


def evaluate(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# ==================== 4. 主函数 ====================

def main():
    """
    Demo主函数：对比监督学习和半监督学习
    """
    print("="*60)
    print("半监督学习Demo - 理解监督+无监督训练原理")
    print("="*60)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-10数据集
    print("\n加载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # ========== 场景设置 ==========
    # 模拟真实场景：只有少量标注数据，大量未标注数据
    total_samples = len(train_dataset)  # 50000
    num_labeled = 1000  # 只有1000个有标签样本（2%）

    print(f"\n数据集设置:")
    print(f"  总训练样本: {total_samples}")
    print(f"  有标签样本: {num_labeled} ({100*num_labeled/total_samples:.1f}%)")
    print(f"  无标签样本: {total_samples - num_labeled} ({100*(total_samples-num_labeled)/total_samples:.1f}%)")

    # 随机选择有标签样本的索引
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    labeled_indices = all_indices[:num_labeled]

    # 创建数据集
    labeled_dataset = SemiSupervisedDataset(train_dataset, labeled_indices, is_labeled=True)
    unlabeled_dataset = SemiSupervisedDataset(train_dataset, labeled_indices, is_labeled=False)

    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True, num_workers=2)

    # 测试集加载器
    test_loader = DataLoader(
        [(img, label) for img, label in test_dataset],
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    # 包装测试集
    test_wrapper = [{'image': img, 'label': label} for img, label in test_dataset]
    test_loader = DataLoader(test_wrapper, batch_size=128, shuffle=False, num_workers=2)

    # ========== 实验1: 纯监督学习（仅用有标签数据） ==========
    print("\n" + "="*60)
    print("实验1: 纯监督学习（仅用1000个有标签样本）")
    print("="*60)

    model_supervised = SimpleCNN(num_classes=10).to(device)
    optimizer_supervised = torch.optim.Adam(model_supervised.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        loss = supervised_train_epoch(model_supervised, labeled_loader, optimizer_supervised, device)
        acc = evaluate(model_supervised, test_loader, device)
        print(f"  平均损失: {loss:.4f} | 测试准确率: {acc:.2f}%")

    final_acc_supervised = evaluate(model_supervised, test_loader, device)
    print(f"\n纯监督学习最终准确率: {final_acc_supervised:.2f}%")

    # ========== 实验2: 半监督学习（有标签+无标签数据） ==========
    print("\n" + "="*60)
    print("实验2: 半监督学习（1000有标签 + 49000无标签样本）")
    print("="*60)

    model_semi = SimpleCNN(num_classes=10).to(device)
    optimizer_semi = torch.optim.Adam(model_semi.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        loss, l_x, l_u = semi_supervised_train_epoch(
            model_semi, labeled_loader, unlabeled_loader,
            optimizer_semi, device, epoch, num_classes=10, lambda_u=10.0
        )
        acc = evaluate(model_semi, test_loader, device)
        print(f"  总损失: {loss:.4f} | 有标签损失: {l_x:.4f} | "
              f"无标签损失: {l_u:.4f} | 测试准确率: {acc:.2f}%")

    final_acc_semi = evaluate(model_semi, test_loader, device)
    print(f"\n半监督学习最终准确率: {final_acc_semi:.2f}%")

    # ========== 结果对比 ==========
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    print(f"纯监督学习准确率: {final_acc_supervised:.2f}%")
    print(f"半监督学习准确率: {final_acc_semi:.2f}%")
    print(f"提升: {final_acc_semi - final_acc_supervised:.2f}%")

    print("\n" + "="*60)
    print("半监督学习的关键原理:")
    print("="*60)
    print("1. 伪标签生成: 用模型预测为无标签数据生成'软标签'")
    print("2. 一致性正则化: 同一样本的不同增强应该有相似的预测")
    print("3. 锐化(Sharpening): 让模型对伪标签更有信心")
    print("4. MixUp: 混合样本增强泛化能力")
    print("5. 平衡损失: L_total = L_labeled + λ * L_unlabeled")
    print("\n关键点: 无标签数据提供了额外的数据分布信息，")
    print("        帮助模型学习更好的特征表示和决策边界。")
    print("="*60)


if __name__ == "__main__":
    main()
