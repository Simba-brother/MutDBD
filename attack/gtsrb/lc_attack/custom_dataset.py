from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, datas, labels, device='cpu'):
        """
        初始化数据集
        
        参数:
            datas: 形状为(B, C, H, W)的图像数据张量
            labels: 形状为(B,)的标签张量
        """
        self.datas = datas.to(device)
        self.labels = labels.to(device)
        
        # 验证数据一致性
        assert len(self.datas) == len(self.labels), "数据量和标签量必须相同"
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.datas)
    
    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        
        参数:
            idx: 索引值
            
        返回:
            图像数据和对应标签
        """
        image = self.datas[idx]
        label = self.labels[idx]
        
        return image,label.item()

# 使用示例
if __name__ == "__main__":
    pass
    # 假设您已经有datas和labels变量
    # datas形状: (B, C, H, W)
    # labels形状: (B,)
    
    # 创建数据集实例
    # dataset = CustomImageDataset(datas, labels)
    
    # # 创建数据加载器
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # # 使用数据加载器进行迭代
    # for batch_idx, (data, target) in enumerate(dataloader):
    #     print(f"批次 {batch_idx}, 数据形状: {data.shape}, 标签形状: {target.shape}")
    #     # 这里可以添加您的训练或处理代码