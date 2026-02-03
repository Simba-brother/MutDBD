'''所有（数据集*攻击）的train_transform和test_transform'''

from torchvision import transforms

def get_cifar10_transform(attack_name):
    if attack_name == "BadNets":
        # 训练集transform    
        train_transform = transforms.Compose([
            transforms.ToPILImage(), # Tensor|ndarray to PIL
            transforms.RandomCrop(size=32,padding=4,padding_mode="reflect"), # PIL|Tensor to be cropped.
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor()
        ])
        # 测试集transform
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    elif attack_name == "IAD":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=5),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
        ])
    elif attack_name == "Refool":
        # 训练集transform
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        # 测试集transform
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
    elif attack_name == "WaNet":
        # 获得训练集transform
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size=32,padding=4,padding_mode="reflect"),
            transforms.RandomHorizontalFlip()
        ])
        # 获得测试集transform
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif attack_name == "LabelConsistent":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif attack_name == "SBA":
        # 训练集transform    
        train_transform = transforms.Compose([
            transforms.ToPILImage(), # Tensor|ndarray to PIL
            transforms.RandomCrop(size=32,padding=4,padding_mode="reflect"), # PIL|Tensor to be cropped.
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor()
        ])
        # 测试集transform
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    else:
        raise ValueError("Invalid input")
    return train_transform,test_transform

def get_gtsrb_transform(attack_name):
    if attack_name == "BadNets":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(size=32,padding=4,padding_mode="reflect"), 
            transforms.ToTensor()
        ])
        # 测试集transform
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    elif attack_name == "IAD":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    elif attack_name == "Refool":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
    elif attack_name == "WaNet":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    elif attack_name == "LabelConsistent":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Invalid input")
    return train_transform,test_transform

def get_imagenet_transform(attack_name):
    if attack_name == "BadNets":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=224), 
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) BadNets不要用
        ])
        # 测试集transform
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) BadNets不要用
        ])
    elif attack_name == "IAD":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomCrop((224, 224), padding=5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif attack_name == "Refool":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
    elif attack_name == "WaNet":
        # 获得训练集transform
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif attack_name == "LabelConsistent":
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Invalid input")
    return train_transform,test_transform
'''
config = read_yaml("config.yaml")
dataset_dir = config["CIFAR10_dataset_dir"]
train_transform,test_transform = get_transform("BadNets")
trainset, test_set = get_dataset(dataset_dir,train_transform,test_transform)
'''