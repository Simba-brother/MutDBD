'''
可视化污染样本img
'''
import torch
import os
from codes import config
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, RandomCrop, Resize,RandomRotation,Normalize
from torchvision.datasets import DatasetFolder
from codes import dataset_transforms_config
import cv2
import pandas as pd


def reverse_normalize(tensor,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    # 创建一个逆Normalize的转换矩阵
    mean = torch.as_tensor(mean).reshape((3, 1, 1))
    std = torch.as_tensor(std).reshape((3, 1, 1))

    return tensor * std + mean

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor,save_path):
    if config.dataset_name == "CIFAR10":
        if config.attack_name == "IAD":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
            tensor = reverse_normalize(tensor,mean,std)
        elif config.attack_name == "Refool":
            tensor = reverse_normalize(tensor)
    elif config.dataset_name == "GTSRB":
        if config.attack_name == "Refool":
            tensor = reverse_normalize(tensor)
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = ToPILImage()(image)
    image.save(save_path)


if __name__ == "__main__":
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    trainset_transform = dataset_transforms_config.config[config.dataset_name][config.attack_name]["trainset"]
    if config.dataset_name == "CIFAR10":
        dataset_dir = config.CIFAR10_dataset_dir
    elif config.dataset_name == "GTSRB":
        dataset_dir = config.GTSRB_dataset_dir
    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=trainset_transform,
        target_transform=None,
        is_valid_file=None)

    # 加载后门数据
    backdoor_data_path = os.path.join(
        config.exp_root_dir, 
        "ATTACK", 
        config.dataset_name, 
        config.model_name, 
        config.attack_name, 
        "backdoor_data.pth")
    # 加载评估数据集
    
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    eval_df = pd.read_csv(os.path.join(
                config.exp_root_dir,
                "EvalMutationToCSV",
                config.dataset_name,
                config.model_name,
                config.attack_name,
                str(0.01),
                "preLabel.csv"
            ))
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_trainset =backdoor_data["poisoned_trainset"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    poisoned_id = list(poisoned_ids)[0]
    # 看clean image
    clean_sample,clean_label = trainset[poisoned_id]
    save_path = "clean_img.png"
    tensor_to_PIL(clean_sample,save_path)
    # 看poisoned image
    poisoned_sample, poisoned_label, isPoisoned =  poisoned_trainset[poisoned_id]
    save_path = "poisoned_img.png"
    tensor_to_PIL(poisoned_sample,save_path)

