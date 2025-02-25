
import os
import cv2
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, RandomCrop, Resize, RandomRotation, Normalize, RandomResizedCrop
from torchvision.datasets import DatasetFolder
from codes import config

# BadNets
def cifar10_BadNets():
    # 训练集transform    
    transform_train = Compose([
        # Convert a tensor or an ndarray to PIL Image
        ToPILImage(), 
        # img (PIL Image or Tensor): Image to be cropped.
        RandomCrop(size=32,padding=4,padding_mode="reflect"), 
        RandomHorizontalFlip(), 
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ToTensor()
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        ToTensor()
    ])

    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    # 数据集文件夹
    testset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread, # 图像加载器
        extensions=('png',), # 图像后缀
        transform=transform_test, # 图像变换器
        target_transform=None, # 图像标签变换器
        is_valid_file=None # 图像验证器
    )
    return trainset,testset

# IAD
def cifar10_IAD():
    # 使用BackdoorBox的transform
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        RandomCrop((32, 32), padding=5),
        RandomRotation(10),
        RandomHorizontalFlip(p=0.5),
        # 图像数据归一化
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        # 归一化
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"), # 文件夹目录
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    # 另外一份训练集
    trainset1 = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"), # 文件夹目录
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # 另外一份测试集
    testset1 = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, trainset1, testset, testset1


# Refool
def cifar10_Refool():
    # 训练集transform
    transform_train = Compose([
        ToPILImage(),
        # RandomCrop(size=32,padding=4,padding_mode="reflect"), 
        Resize((32, 32)),
        RandomHorizontalFlip(p=1),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"), # 文件夹目录
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset

def cifar10_WaNet():
    # 获得训练集transform
    transform_train = Compose([
        ToTensor(),
        RandomCrop(size=32,padding=4,padding_mode="reflect"),
        RandomHorizontalFlip()
    ])
    # 获得测试集transform
    transform_test = Compose([
        ToTensor()
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"), # 文件夹目录
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "test"), # 文件夹目录
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset

def gtsrb_BadNets():
    # 训练集transform    
    transform_train = Compose([
        ToPILImage(),
        RandomCrop(size=32,padding=4,padding_mode="reflect"), 
        ToTensor()
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])

    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.GTSRB_dataset_dir,"train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset

def gtsrb_IAD():
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        RandomCrop((32, 32), padding=5),
        RandomRotation(10),
        ToTensor()
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    trainset1 = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # 另外一份测试集
    testset1 = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,trainset1,testset,testset1

def gtsrb_Refool():
    # 训练集transform
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset

def gtsrb_WaNet():
    # 获得训练集transform
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    transform_test = Compose([
        ToTensor(),
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.GTSRB_dataset_dir,"test"),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,testset

'''
ImageNet
'''
# BadNets
def imagenet_BadNets():
    # 训练集transform    
    transform_train = Compose([
        ToPILImage(),
        RandomResizedCrop(size=224), 
        ToTensor()
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])

    # 获得数据集
    trainset = DatasetFolder(
        root= os.path.join(config.ImageNet2012_subset_dir,"train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"test"),
        loader=cv2.imread,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset, testset

# IAD
def imagenet_IAD():
    transform_train = Compose([
        ToPILImage(),
        Resize((224, 224)),
        RandomCrop((224, 224), padding=5),
        RandomRotation(10),
        ToTensor()
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    # 另外一份训练集
    trainset1 = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"test"),
        loader=cv2.imread,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # 另外一份测试集
    testset1 = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"test"),
        loader=cv2.imread,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,trainset1,testset,testset1

# Refool
def imagenet_Refool():
    # 训练集transform
    transform_train = Compose([
        ToPILImage(),
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"test"),
        loader=cv2.imread,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,testset

# WaNet
def imagenet_WaNet():
    # 获得训练集transform
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    transform_test = Compose([
        ToTensor(),
        ToPILImage(),
        Resize((224, 224)),
        ToTensor()
    ])
    # 获得数据集
    trainset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"train"),
        loader=cv2.imread, # ndarray
        extensions=('jpeg',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root=os.path.join(config.ImageNet2012_subset_dir,"test"),
        loader=cv2.imread,
        extensions=('jpeg',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    return trainset,testset