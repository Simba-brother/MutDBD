import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,Subset,ConcatDataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, RandomCrop, Resize, RandomRotation, Normalize, RandomResizedCrop
from codes import config
from codes.common.eval_model import EvalModel
from codes.look_poisoned_img import reverse_normalize
# cifar10
from codes.poisoned_dataset.cifar10.BadNets.generator import gen_poisoned_dataset as cifar10_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.IAD.generator import gen_poisoned_dataset as cifar10_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.Refool.generator import gen_poisoned_dataset as cifar10_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.cifar10.WaNet.generator import gen_poisoned_dataset as cifar10_WaNet_gen_poisoned_dataset
# gtsrb
from codes.poisoned_dataset.gtsrb.BadNets.generator import gen_poisoned_dataset as gtsrb_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.IAD.generator import gen_poisoned_dataset as gtsrb_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.Refool.generator import gen_poisoned_dataset as gtsrb_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.gtsrb.WaNet.generator import gen_poisoned_dataset as gtsrb_WaNet_gen_poisoned_dataset
# imagenet
from codes.poisoned_dataset.imagenet_sub.BadNets.generator import gen_poisoned_dataset as imagenet_badNets_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.IAD.generator import gen_poisoned_dataset as imagenet_IAD_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.Refool.generator import gen_poisoned_dataset as imagenet_Refool_gen_poisoned_dataset
from codes.poisoned_dataset.imagenet_sub.WaNet.generator import gen_poisoned_dataset as imagenet_WaNet_gen_poisoned_dataset

from codes.transform_dataset import cifar10_IAD,cifar10_BadNets
class MyDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        sample,label = self.dataset[index]
        sample = self.transforms(sample)
        return sample,label
    
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset_name = "CIFAR10"
    model_name = "ResNet18"
    attack_name = "IAD"

    backdoor_data = torch.load(os.path.join(config.exp_root_dir, "ATTACK", 
                                            dataset_name, 
                                            model_name, 
                                            attack_name, "backdoor_data.pth"), map_location="cpu")
    backdoor_model = backdoor_data["backdoor_model"]
    poisoned_ids = backdoor_data["poisoned_ids"]
    
    poisoned_ids = list(poisoned_ids)
    
    poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
    poisoned_testset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"test")
    poisoned_testset_fixed = backdoor_data["poisoned_testset"]

    clean_trainset, _, clean_testset, _ = cifar10_IAD()

    device = torch.device("cuda:0")
    em = EvalModel(backdoor_model, poisoned_trainset, device)
    acc = em.eval_acc()
    print(f"poisoned_trainset:{acc}")

    em = EvalModel(backdoor_model, clean_testset, device)
    acc = em.eval_acc()
    print(f"clean_testset:{acc}")

    em = EvalModel(backdoor_model, poisoned_testset, device)
    acc = em.eval_acc()
    print(f"poisoned_testset:{acc}")

    em = EvalModel(backdoor_model, poisoned_testset_fixed, device)
    acc = em.eval_acc()
    print(f"poisoned_testset_fixed:{acc}")

    pure_poisoned_trainset = Subset(poisoned_trainset,poisoned_ids)
    em = EvalModel(backdoor_model, pure_poisoned_trainset, device)
    old_acc = em.eval_acc()
    print(f"pure_poisoned_trainset:{old_acc}")

    _dataset = []
    _target_list = []
    for poisoned_id in poisoned_ids:
        sample, target, isPoisoned = poisoned_trainset[poisoned_id]
        sample = reverse_normalize(sample,mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        # sample = ToPILImage()(sample)
        _dataset.append((sample,target))
        _target_list.append(target)

    transforms_train = Compose([
        ToPILImage(),
        # Resize((32, 32)),
        # RandomCrop((32, 32), padding=5),
        RandomRotation(10),
        # RandomHorizontalFlip(p=0.5),
        # 图像数据归一化
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])

    myDataset = MyDataset(_dataset, transforms_train)
    em = EvalModel(backdoor_model, myDataset, device)
    new_acc = em.eval_acc() 
    print(f"myDataset:{new_acc}")
    print(_target_list)

