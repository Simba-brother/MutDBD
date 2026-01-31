"""
半监督学习简化Demo - 核心原理演示

这个脚本用最简单的方式展示半监督学习的核心思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class SimpleNN(nn.Module):
    """简单的2层神经网络"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def visualize_decision_boundary(model, X, y_labeled, labeled_mask, title, save_path=None):
    """可视化决策边界"""
    model.eval()

    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = F.softmax(Z, dim=1)[:, 1].numpy()
    Z = Z.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, levels=20, cmap='RdYlBu')
    plt.colorbar(label='Class 1 Probability')

    # 绘制有标签样本（大圆点）
    labeled_indices = np.where(labeled_mask)[0]
    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1],
                c=y_labeled[labeled_indices], s=100, edgecolors='black',
                linewidths=2, cmap='RdYlBu', label='Labeled', marker='o')

    # 绘制无标签样本（小圆点）
    unlabeled_indices = np.where(~labeled_mask)[0]
    plt.scatter(X[unlabeled_indices, 0], X[unlabeled_indices, 1],
                c='gray', s=30, alpha=0.3, label='Unlabeled', marker='.')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    plt.close()


def supervised_train(model, X_labeled, y_labeled, epochs=100, lr=0.01):
    """纯监督训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_labeled)
    y_tensor = torch.LongTensor(y_labeled)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def semi_supervised_train(model, X_all, y_labeled, labeled_mask,
                          epochs=100, lr=0.01, lambda_u=1.0):
    """
    半监督训练

    核心思想：
    1. 有标签数据：用真实标签计算交叉熵损失
    2. 无标签数据：用模型自己的预测作为"伪标签"，计算一致性损失
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_all)
    y_tensor = torch.LongTensor(y_labeled)

    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(~labeled_mask)[0]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # ===== 有标签数据的监督损失 =====
        X_labeled = X_tensor[labeled_indices]
        y_labeled_batch = y_tensor[labeled_indices]
        outputs_labeled = model(X_labeled)
        supervised_loss = criterion(outputs_labeled, y_labeled_batch)

        # ===== 无标签数据的一致性损失 =====
        X_unlabeled = X_tensor[unlabeled_indices]

        # 生成伪标签（用模型当前的预测）
        with torch.no_grad():
            pseudo_logits = model(X_unlabeled)
            pseudo_probs = F.softmax(pseudo_logits, dim=1)

            # 锐化：让模型更有信心
            pseudo_probs = pseudo_probs ** 2  # 简化版的锐化
            pseudo_probs = pseudo_probs / pseudo_probs.sum(dim=1, keepdim=True)

        # 让模型的预测接近伪标签
        outputs_unlabeled = model(X_unlabeled)
        probs_unlabeled = F.softmax(outputs_unlabeled, dim=1)

        # 一致性损失：MSE
        consistency_loss = torch.mean((probs_unlabeled - pseudo_probs) ** 2)

        # ===== 总损失 =====
        total_loss = supervised_loss + lambda_u * consistency_loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Total Loss: {total_loss.item():.4f}, "
                  f"Supervised: {supervised_loss.item():.4f}, "
                  f"Consistency: {consistency_loss.item():.4f}")


def main():
    """主函数：对比监督学习和半监督学习"""
    print("="*70)
    print("半监督学习简化Demo - 可视化决策边界")
    print("="*70)

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # ===== 生成数据 =====
    print("\n生成月牙形数据集...")
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

    # 模拟真实场景：只有少量标注数据
    n_labeled = 30  # 只有30个有标签样本（10%）
    labeled_indices = np.random.choice(len(X), n_labeled, replace=False)
    labeled_mask = np.zeros(len(X), dtype=bool)
    labeled_mask[labeled_indices] = True

    print(f"总样本数: {len(X)}")
    print(f"有标签样本: {n_labeled} ({100*n_labeled/len(X):.1f}%)")
    print(f"无标签样本: {len(X) - n_labeled} ({100*(len(X)-n_labeled)/len(X):.1f}%)")

    # ===== 实验1: 纯监督学习 =====
    print("\n" + "="*70)
    print("实验1: 纯监督学习（仅用30个有标签样本）")
    print("="*70)

    model_supervised = SimpleNN()
    X_labeled = X[labeled_mask]
    y_labeled_only = y[labeled_mask]

    supervised_train(model_supervised, X_labeled, y_labeled_only, epochs=100, lr=0.01)

    # 评估
    with torch.no_grad():
        outputs = model_supervised(torch.FloatTensor(X))
        _, predicted = torch.max(outputs, 1)
        acc_supervised = (predicted.numpy() == y).mean() * 100
    print(f"\n纯监督学习准确率: {acc_supervised:.2f}%")

    # 可视化
    visualize_decision_boundary(
        model_supervised, X, y, labeled_mask,
        f"纯监督学习 (准确率: {acc_supervised:.2f}%)",
        save_path="/home/mml/workspace/backdoor_detect/supervised_boundary.png"
    )

    # ===== 实验2: 半监督学习 =====
    print("\n" + "="*70)
    print("实验2: 半监督学习（30有标签 + 270无标签样本）")
    print("="*70)

    model_semi = SimpleNN()

    # 创建完整的标签数组（无标签位置用-1填充，但实际不会用到）
    y_full = y.copy()

    semi_supervised_train(
        model_semi, X, y_full, labeled_mask,
        epochs=100, lr=0.01, lambda_u=2.0
    )

    # 评估
    with torch.no_grad():
        outputs = model_semi(torch.FloatTensor(X))
        _, predicted = torch.max(outputs, 1)
        acc_semi = (predicted.numpy() == y).mean() * 100
    print(f"\n半监督学习准确率: {acc_semi:.2f}%")

    # 可视化
    visualize_decision_boundary(
        model_semi, X, y, labeled_mask,
        f"半监督学习 (准确率: {acc_semi:.2f}%)",
        save_path="/home/mml/workspace/backdoor_detect/semi_supervised_boundary.png"
    )

    # ===== 结果对比 =====
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    print(f"纯监督学习准确率: {acc_supervised:.2f}%")
    print(f"半监督学习准确率: {acc_semi:.2f}%")
    print(f"提升: {acc_semi - acc_supervised:.2f}%")

    print("\n" + "="*70)
    print("关键观察:")
    print("="*70)
    print("1. 查看生成的图像，观察决策边界的差异")
    print("2. 半监督学习利用了无标签数据的分布信息")
    print("3. 无标签数据帮助模型学习更平滑、更合理的决策边界")
    print("4. 当有标签数据很少时，半监督学习的优势更明显")
    print("="*70)

    print("\n可视化图像已保存:")
    print("  - supervised_boundary.png (纯监督学习)")
    print("  - semi_supervised_boundary.png (半监督学习)")


if __name__ == "__main__":
    main()
