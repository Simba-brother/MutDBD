'''
可视化污染样本img
'''
import torch
import os
from codes import config
from torchvision import transforms
from torchvision.transforms import CenterCrop
from torchvision.datasets import DatasetFolder
import cv2
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

# transform数据集
from codes.transform_dataset import cifar10_BadNets, cifar10_IAD, cifar10_Refool, cifar10_WaNet
from codes.transform_dataset import gtsrb_BadNets, gtsrb_IAD, gtsrb_Refool, gtsrb_WaNet
from codes.transform_dataset import imagenet_BadNets, imagenet_IAD, imagenet_Refool, imagenet_WaNet


def get_fresh_dataset(poisoned_ids):
    if dataset_name == "CIFAR10":
        if attack_name == "BadNets":
            poisoned_trainset = cifar10_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = cifar10_IAD_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = cifar10_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = cifar10_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = cifar10_WaNet_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, clean_testset = cifar10_WaNet()
    elif dataset_name == "GTSRB":
        if attack_name == "BadNets":
            poisoned_trainset = gtsrb_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = gtsrb_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = gtsrb_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = gtsrb_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = gtsrb_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = gtsrb_WaNet()
    elif dataset_name == "ImageNet2012_subset":
        if attack_name == "BadNets":
            poisoned_trainset = imagenet_badNets_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_BadNets()
        elif attack_name == "IAD":
            poisoned_trainset = imagenet_IAD_gen_poisoned_dataset(model_name,poisoned_ids,"train")
            clean_trainset, _, clean_testset, _ = imagenet_IAD()
        elif attack_name == "Refool":
            poisoned_trainset = imagenet_Refool_gen_poisoned_dataset(poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_Refool()
        elif attack_name == "WaNet":
            poisoned_trainset = imagenet_WaNet_gen_poisoned_dataset(model_name, poisoned_ids,"train")
            clean_trainset, clean_testset = imagenet_WaNet()
    return poisoned_trainset, clean_trainset, clean_testset


def reverse_normalize(tensor,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    # 创建一个逆Normalize的转换矩阵
    mean = torch.as_tensor(mean).reshape((3, 1, 1))
    std = torch.as_tensor(std).reshape((3, 1, 1))

    return tensor * std + mean

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    img = tensor.cpu().clone()
    img = img.squeeze(0) # 压缩一维
    img = transforms.ToPILImage()(img)
    return img


def look_CIFAR10():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    dataset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    
    poisoned_ids = [1,5,10,50,150,200,250,300,1000,3000] 
    _id = 3000

    # Clean
    clean_img = dataset[_id][0]
    clean_img = tensor_to_PIL(clean_img)
    clean_img.save(f"imgs/cases/{dataset_name}/clean_img.png")

    # BadNets
    from codes.poisoned_dataset.cifar10.BadNets.generator import PoisonedDatasetFolder as BadNetsDataset
    # backdoor pattern
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255 # 用于归一化前
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    badnets_dataset = BadNetsDataset(dataset,attack_class,poisoned_ids,pattern,weight,-1,0) # ToTenser前
    badnets_img = badnets_dataset[_id][0]
    badnets_img = tensor_to_PIL(badnets_img)
    badnets_img.save(f"imgs/cases/{dataset_name}/badnets_img.png")

    # IAD
    from codes.poisoned_dataset.cifar10.IAD.generator import IADPoisonedDatasetFolder as IADDataset
    from codes.poisoned_dataset.cifar10.IAD.generator import get_attack_dict_path as get_iad_attack_dict_path
    from codes.core.attacks.IAD import Generator
    iad_attack_dict_path = get_iad_attack_dict_path(model_name)
    modelG = Generator("cifar10")
    modelM = Generator("cifar10", out_channels=1)
    dict_state = torch.load(iad_attack_dict_path, map_location="cpu")
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])
    modelG.eval()
    modelM.eval()
    iad_dataset = IADDataset(dataset,attack_class,poisoned_ids,modelG,modelM)
    iad_img = iad_dataset[_id][0]
    iad_img = tensor_to_PIL(iad_img)
    iad_img.save(f"imgs/cases/{dataset_name}/iad_img.png")

    # Refool
    from codes.poisoned_dataset.cifar10.Refool.generator import PoisonedDatasetFolder as RefoolDataset
    from codes.poisoned_dataset.cifar10.Refool.generator import read_image
    reflection_images = []
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    refool_dataset = RefoolDataset(dataset,attack_class,poisoned_ids, 
            poisoned_transform_index=-1,  # PIL后
            poisoned_target_transform_index=0, 
            reflection_cadidates=reflection_images,
            max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.)
    refool_img = refool_dataset[_id][0]
    refool_img = tensor_to_PIL(refool_img)
    refool_img.save(f"imgs/cases/{dataset_name}/refool_img.png")

    # WaNet
    from codes.poisoned_dataset.cifar10.WaNet.generator import PoisonedDatasetFolder as WaNetDataset
    from codes.poisoned_dataset.cifar10.WaNet.generator import get_attack_dict_path as get_wanet_attack_dict_path
    attack_dict_path = get_wanet_attack_dict_path(model_name)
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    # trigger
    identity_grid = dict_state["identity_grid"]
    noise_grid = dict_state["noise_grid"]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = DatasetFolder(
        root= os.path.join(config.CIFAR10_dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    wanet_dataset= WaNetDataset(dataset,attack_class,poisoned_ids,identity_grid,noise_grid,noise=False,
                                poisoned_transform_index=0, # 最前
                                poisoned_target_transform_index=0)
    wanet_img = wanet_dataset[_id][0]
    wanet_img = tensor_to_PIL(wanet_img)
    wanet_img.save(f"imgs/cases/{dataset_name}/wanet_img.png")

    print("END")

def look_GTSRB():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])
    dataset = DatasetFolder(
        root= os.path.join(config.GTSRB_dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    
    poisoned_ids = [1,5,10,50,150,200,250,300,1000,3000] 
    _id = 3000

    # Clean
    clean_img = dataset[_id][0]
    clean_img = tensor_to_PIL(clean_img)
    clean_img.save(f"imgs/cases/{dataset_name}/clean_img.png")

    # BadNets
    from codes.poisoned_dataset.gtsrb.BadNets.generator import PoisonedDatasetFolder as BadNetsDataset
    # backdoor pattern
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255 # 用于归一化前
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    badnets_dataset = BadNetsDataset(dataset,attack_class,poisoned_ids,pattern,weight,-1,0) # ToTenser前
    badnets_img = badnets_dataset[_id][0]
    badnets_img = tensor_to_PIL(badnets_img)
    badnets_img.save(f"imgs/cases/{dataset_name}/badnets_img.png")

    # IAD
    from codes.poisoned_dataset.gtsrb.IAD.generator import IADPoisonedDatasetFolder as IADDataset
    from codes.poisoned_dataset.gtsrb.IAD.generator import get_attack_dict_path as get_iad_attack_dict_path
    from codes.core.attacks.IAD import Generator
    iad_attack_dict_path = get_iad_attack_dict_path(model_name)
    modelG = Generator("gtsrb")
    modelM = Generator("gtsrb", out_channels=1)
    dict_state = torch.load(iad_attack_dict_path, map_location="cpu")
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])
    modelG.eval()
    modelM.eval()
    iad_dataset = IADDataset(dataset,attack_class,poisoned_ids,modelG,modelM)
    iad_img = iad_dataset[_id][0]
    iad_img = tensor_to_PIL(iad_img)
    iad_img.save(f"imgs/cases/{dataset_name}/iad_img.png")

    # Refool
    from codes.poisoned_dataset.gtsrb.Refool.generator import PoisonedDatasetFolder as RefoolDataset
    from codes.poisoned_dataset.gtsrb.Refool.generator import read_image
    reflection_images = []
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    refool_dataset = RefoolDataset(dataset,attack_class,poisoned_ids, 
            poisoned_transform_index=-1,  # PIL后
            poisoned_target_transform_index=0, 
            reflection_cadidates=reflection_images,
            max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.)
    refool_img = refool_dataset[_id][0]
    refool_img = tensor_to_PIL(refool_img)
    refool_img.save(f"imgs/cases/{dataset_name}/refool_img.png")

    # WaNet
    from codes.poisoned_dataset.gtsrb.WaNet.generator import PoisonedDatasetFolder as WaNetDataset
    from codes.poisoned_dataset.gtsrb.WaNet.generator import get_attack_dict_path as get_wanet_attack_dict_path
    attack_dict_path = get_wanet_attack_dict_path(model_name)
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    # trigger
    identity_grid = dict_state["identity_grid"]
    noise_grid = dict_state["noise_grid"]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = DatasetFolder(
        root= os.path.join(config.GTSRB_dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    wanet_dataset= WaNetDataset(dataset,attack_class,poisoned_ids,identity_grid,noise_grid,noise=False,
                                poisoned_transform_index=0, # 最前
                                poisoned_target_transform_index=0)
    wanet_img = wanet_dataset[_id][0]
    wanet_img = tensor_to_PIL(wanet_img)
    wanet_img.save(f"imgs/cases/{dataset_name}/wanet_img.png")
    print("END")

def look_ImageNet():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    dataset = DatasetFolder(
        root= os.path.join(config.ImageNet2012_subset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('jpeg',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    
    poisoned_ids = [18000] # [666,1000,3000,4500,6000,8000,12000,15000,18000]
    


    # BadNets
    from codes.poisoned_dataset.imagenet_sub.BadNets.generator import PoisonedDatasetFolder as BadNetsDataset
    # backdoor pattern
    pattern = torch.zeros((224, 224), dtype=torch.uint8)
    pattern[-26:-4, -26:-4] = 255 # 用于归一化前
    weight = torch.zeros((224, 224), dtype=torch.float32)
    weight[-26:-4, -26:-4] = 1.0
    badnets_dataset = BadNetsDataset(dataset,attack_class,poisoned_ids,pattern,weight,-1,0) # ToTenser前


    # IAD
    from codes.poisoned_dataset.imagenet_sub.IAD.generator import IADPoisonedDatasetFolder as IADDataset
    from codes.core.attacks.IAD import Generator

    backdoor_data = os.path.join(
        config.exp_root_dir,
        "ATTACK",
        "ImageNet2012_subset",
        f"{model_name}",
        "IAD",
        "backdoor_data.pth"
    )
    modelG = Generator("ImageNet")
    modelM = Generator("ImageNet", out_channels=1)
    backdoor_data = torch.load(backdoor_data, map_location="cpu")
    # # 在数据集转换组合transforms.Compose[]的最后进行中毒植入
    modelG.load_state_dict(backdoor_data["modelG"])
    modelM.load_state_dict(backdoor_data["modelM"])
    modelG.eval()
    modelM.eval()
    iad_dataset = IADDataset(dataset,attack_class,poisoned_ids,modelG,modelM)


    # Refool
    from codes.poisoned_dataset.imagenet_sub.Refool.generator import PoisonedDatasetFolder as RefoolDataset
    from codes.poisoned_dataset.imagenet_sub.Refool.generator import read_image
    reflection_images = []
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    refool_dataset = RefoolDataset(dataset,attack_class,poisoned_ids, 
            poisoned_transform_index=-1,  # PIL后
            poisoned_target_transform_index=0, 
            reflection_cadidates=reflection_images,
            max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.)


    # WaNet
    from codes.poisoned_dataset.imagenet_sub.WaNet.generator import PoisonedDatasetFolder as WaNetDataset
    backdoor_data_path = os.path.join(
        config.exp_root_dir,
        "ATTACK",
        "ImageNet2012_subset",
        f"{model_name}",
        "WaNet",
        "backdoor_data.pth"
    )
    backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    # trigger
    identity_grid = backdoor_data["identity_grid"]
    noise_grid = backdoor_data["noise_grid"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])
    dataset_other = DatasetFolder(
        root= os.path.join(config.ImageNet2012_subset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('jpeg',),
        transform=transform,
        target_transform=None,
        is_valid_file=None)
    wanet_dataset= WaNetDataset(dataset_other,attack_class,poisoned_ids,identity_grid,noise_grid,noise=False,
                                poisoned_transform_index=0, # 最前
                                poisoned_target_transform_index=0)


    for _id in poisoned_ids:
        save_dir = f"imgs/cases/{dataset_name}/{_id}"
        os.makedirs(save_dir,exist_ok=True)
        # Clean
        clean_img = dataset[_id][0]
        clean_img = tensor_to_PIL(clean_img)
        clean_img.save(os.path.join(save_dir,"clean.png"))
        # BadNets
        badnets_img = badnets_dataset[_id][0]
        badnets_img = tensor_to_PIL(badnets_img)
        badnets_img.save(os.path.join(save_dir,"badnets.png"))
        # IAD
        iad_img = iad_dataset[_id][0]
        iad_img = tensor_to_PIL(iad_img)
        iad_img.save(os.path.join(save_dir,"iad.png"))
        # Refool
        refool_img = refool_dataset[_id][0]
        refool_img = tensor_to_PIL(refool_img)
        refool_img.save(os.path.join(save_dir,"refool.png"))
        # WaNet
        wanet_img = wanet_dataset[_id][0]
        wanet_img = tensor_to_PIL(wanet_img)
        wanet_img.save(os.path.join(save_dir,"wanet.png"))

    print("END")

if __name__ == "__main__":
    dataset_name = "ImageNet2012_subset"
    attack_name = "BadNets"
    model_name = "ResNet18"
    attack_class = 3
    # look_GTSRB()
    look_ImageNet()
    
    # 加载后门数据
    # backdoor_data_path = os.path.join(
    #     config.exp_root_dir, 
    #     "ATTACK", 
    #     dataset_name, 
    #     model_name, 
    #     attack_name, 
    #     "backdoor_data.pth")
    # backdoor_data = torch.load(backdoor_data_path, map_location="cpu")
    # 加载评估数据集
    # eval_df = pd.read_csv(os.path.join(
    #             config.exp_root_dir,
    #             "EvalMutationToCSV",
    #             config.dataset_name,
    #             config.model_name,
    #             config.attack_name,
    #             str(0.01),
    #             "preLabel.csv"
    #         ))
    # poisoned_ids = backdoor_data["poisoned_ids"]
    # poisoned_testset = backdoor_data["poisoned_testset"]



    

