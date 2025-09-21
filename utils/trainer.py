import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional, Callable, Union
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
    """
    一个通用的PyTorch神经网络训练类
    
    功能包括:
    - 模型训练与验证
    - 学习率调度
    - 早停机制
    - 模型保存与加载
    - 训练过程可视化
    - 训练指标记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        optimizer: optim.Optimizer = None,
        criterion: nn.Module = None,
        init_lr: float = 0.01,
        lr_scheduler: optim.lr_scheduler._LRScheduler = None,
        model_dir: str = "models",
        experiment_name: str = "experiment"
    ):
        """
        初始化训练器
        
        参数:
            model: 要训练的神经网络模型
            device: 训练设备 (CPU/GPU), 默认为自动选择
            optimizer: 优化器, 默认为Adam
            criterion: 损失函数, 默认为交叉熵损失
            lr_scheduler: 学习率调度器, 默认为None
            model_dir: 模型保存目录
            experiment_name: 实验名称, 用于创建子目录
        """
        self.model = model
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # 设置默认优化器和损失函数
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=init_lr)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler
        
        # 创建模型保存目录
        self.model_dir = Path(model_dir) / experiment_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 最佳模型参数
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        print(f"使用设备: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            epoch_loss: 平均训练损失
            epoch_acc: 平均训练准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            data = batch[0].to(self.device)
            target = batch[1].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                print(f'训练批次: {batch_idx}/{len(train_loader)}, '
                      f'损失: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        在验证集上评估模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = None,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值, 默认为None(不使用早停)
            save_best: 是否保存最佳模型
            
        返回:
            history: 训练历史记录
        """
        start_time = time.time()
        patience_counter = 0
        
        print(f"开始训练, 共 {epochs} 个epoch...")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler:
                # 根据验证损失调整学习率
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss and save_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                self.save_model(f"best_model_epoch_{epoch}.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start
            print(f'Epoch: {epoch}/{epochs} | 时间: {epoch_time:.2f}s | '
                  f'学习率: {current_lr:.6f} | '
                  f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
                  f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')
            
            # 早停检查
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"早停于第 {epoch} 个epoch, 最佳模型在第 {self.best_epoch} 个epoch")
                break
        
        # 训练完成，加载最佳模型
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"已加载第 {self.best_epoch} 个epoch的最佳模型")
        
        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.2f}s")
        
        return self.history
    
    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用训练好的模型进行预测
        
        参数:
            data_loader: 数据加载器
            
        返回:
            predictions: 预测结果
            true_labels: 真实标签
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.append(predicted.cpu())
                all_labels.append(target)
        
        return torch.cat(all_predictions), torch.cat(all_labels)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估模型性能
        
        参数:
            test_loader: 测试数据加载器
            
        返回:
            metrics: 评估指标字典
        """
        predictions, true_labels = self.predict(test_loader)
        accuracy = (predictions == true_labels).float().mean().item()
        
        # 计算其他指标，如精确率、召回率、F1分数等
        # 这里可以根据需要添加更多指标计算
        
        return {
            'accuracy': accuracy,
            # 可以添加更多指标
        }
    
    def save_model(self, filename: str) -> None:
        """
        保存模型和训练状态
        
        参数:
            filename: 保存的文件名
        """
        save_path = self.model_dir / filename
        
        # 保存模型状态、优化器状态和训练历史
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'loss': self.best_val_loss,
            'history': self.history
        }, save_path)
        
        # 同时保存训练历史为JSON文件
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # 将numpy数组转换为列表以便JSON序列化
            json_history = {k: [float(x) for x in v] for k, v in self.history.items()}
            json.dump(json_history, f, indent=4)
        
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, filename: str) -> None:
        """
        加载模型和训练状态
        
        参数:
            filename: 要加载的文件名
        """
        load_path = self.model_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.lr_scheduler and checkpoint['scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['loss']
        self.history = checkpoint['history']
        
        print(f"模型已从 {load_path} 加载, 最佳epoch: {self.best_epoch}, 最佳验证损失: {self.best_val_loss:.4f}")
    
    def plot_training_history(self, save_plot: bool = True) -> None:
        """
        绘制训练历史图表
        
        参数:
            save_plot: 是否保存图表
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(self.history['train_loss'], label='训练损失')
        ax1.plot(self.history['val_loss'], label='验证损失')
        ax1.set_title('模型损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(self.history['train_acc'], label='训练准确率')
        ax2.plot(self.history['val_acc'], label='验证准确率')
        ax2.set_title('模型准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.model_dir / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图表已保存到: {plot_path}")
        
        plt.show()
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """
        获取学习曲线数据
        
        返回:
            学习曲线数据字典
        """
        return self.history.copy()

# 使用示例
if __name__ == "__main__":
    # 示例用法
    from torchvision import datasets, transforms
    from torchvision.models import resnet18
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载数据集 (示例使用MNIST)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型 (示例使用ResNet18)
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以适应MNIST的单通道输入
    model.fc = nn.Linear(model.fc.in_features, 10)  # 修改最后一层以适应10个类别
    
    # 创建训练器
    trainer = NeuralNetworkTrainer(
        model=model,
        experiment_name="mnist_resnet18"
    )
    
    # 训练模型
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,  # 这里用测试集作为验证集，实际应用中应使用单独的验证集
        epochs=10,
        early_stopping_patience=3,
        save_best=True
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 评估模型
    metrics = trainer.evaluate(test_loader)
    print(f"测试集准确率: {metrics['accuracy'] * 100:.2f}%")
    
    # 保存最终模型
    trainer.save_model("final_model.pth")