from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, RandomCrop, Resize,RandomRotation,Normalize
from torchvision.datasets import DatasetFolder
import cv2
config = {
    "CIFAR10":{
        "BadNets":{
            "trainset":Compose([
                        # Convert a tensor or an ndarray to PIL Image
                        ToPILImage(), 
                        # img (PIL Image or Tensor): Image to be cropped.
                        RandomCrop(size=32,padding=4,padding_mode="reflect"), 
                        RandomHorizontalFlip(), 
                        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                        ToTensor()]),
            "testset":Compose([
                        ToPILImage(),
                        ToTensor()])
        },
        "IAD":{
            "trainset":Compose([
                        ToPILImage(),
                        Resize((32, 32)),
                        RandomCrop((32, 32), padding=5),
                        RandomRotation(10),
                        RandomHorizontalFlip(p=0.5),
                        ToTensor(),
                        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))]),
            "testset":Compose([
                        ToPILImage(),
                        Resize((32, 32)),
                        ToTensor(),
                        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))])
        },
        "Refool":{
            "trainset":Compose([
                        ToPILImage(),
                        Resize((32, 32)),
                        RandomHorizontalFlip(p=1),
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
            "testset":Compose([
                        ToPILImage(),
                        Resize((32, 32)),
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        },
        "WaNet":{
            "trainset":Compose([
                        ToTensor(),
                        RandomCrop(size=32,padding=4,padding_mode="reflect"),
                        RandomHorizontalFlip()]),
            "testset":Compose([ToTensor()])
        }
    },
    "GTSRN":{
        "BadNets":{
            "trainset":Compose([
                        ToPILImage(),
                        RandomCrop(size=32,padding=4,padding_mode="reflect"), 
                        ToTensor()
                    ]),
            "testset":Compose([
                        ToPILImage(),
                        Resize((32, 32)),
                        ToTensor()
                    ])
        },
        "IAD":{
            "trainset":Compose([
                ToPILImage(),
                Resize((32, 32)),
                RandomCrop((32, 32), padding=5),
                RandomRotation(10),
                ToTensor()]),
            "testset":Compose([
                    ToPILImage(),
                    Resize((32, 32)),
                    ToTensor()])
        },
        "Refool":{
            "trainet":Compose([
                    ToPILImage(),
                    Resize((32, 32)),
                    RandomHorizontalFlip(p=0.5),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]),
            "testset":Compose([
                    ToPILImage(),
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        },
        "WaNet":{
            "trainset":Compose([
                    ToTensor(),
                    RandomHorizontalFlip(),
                    ToPILImage(),
                    Resize((32, 32)),
                    ToTensor()]),
            "testset":Compose([
                    ToTensor(),
                    ToPILImage(),
                    Resize((32, 32)),
                    ToTensor()])
        }
    }
}