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


def reverse_normalize(tensor):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]

    # 创建一个逆Normalize的转换矩阵
    mean = torch.as_tensor(mean).reshape((3, 1, 1))
    std = torch.as_tensor(std).reshape((3, 1, 1))

    return tensor * std + mean

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor,save_path):
    if config.dataset_name == "CIFAR10" and config.attack_name in ["IAD","Refool"]:
        tensor = reverse_normalize(tensor)
    elif config.dataset_name == "GTSRB" and config.attack_name in ["Refool"]:
        tensor = reverse_normalize(tensor)
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = ToPILImage()(image)
    image.save(save_path)


if __name__ == "__main__":
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    trainset_transform = dataset_transforms_config[config.dataset_name][config.attack_name]["trainset"]
    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"),
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
    print(f"{config.dataset_name}|{config.model_name}|{config.attack_name}")
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
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

