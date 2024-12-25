import torch
from codes.core.attacks.IAD import Generator 
from codes.scripts.dataset_constructor import IADPoisonedDatasetFolder,ExtractDataset
from codes import config

def create_backdoor_data(attack_dict_path,model,clean_trainset,clean_testset, save_path):
    
    dict_state = torch.load(attack_dict_path, map_location="cpu")
    model.load_state_dict(dict_state["model"])
    backdoor_model = model

    modelG = Generator("cifar10")
    modelM = Generator("cifar10", out_channels=1)
    
    modelG.load_state_dict(dict_state["modelG"])
    modelM.load_state_dict(dict_state["modelM"])

    backdoor_model.eval()
    modelG.eval()
    modelM.eval()

    poisoned_trainset =  IADPoisonedDatasetFolder(
        benign_dataset = clean_trainset,
        y_target = config.target_class_idx,
        poisoned_rate = config.poisoned_rate,
        modelG = modelG,
        modelM =modelM
    )
    
    poisoned_ids = poisoned_trainset.poisoned_set

    poisoned_testset =  IADPoisonedDatasetFolder(
        benign_dataset = clean_testset,
        y_target = config.target_class_idx,
        poisoned_rate = 1,
        modelG = modelG,
        modelM = modelM
    )
    
    # 将数据集抽取到内存，为了加速评估
    poisoned_trainset = ExtractDataset(poisoned_trainset)
    poisoned_testset = ExtractDataset(poisoned_testset)
    
    backdoor_data = {
        "backdoor_model":backdoor_model,
        "poisoned_trainset":poisoned_trainset,
        "poisoned_testset":poisoned_testset,
        "clean_testset":clean_testset,
        "poisoned_ids":poisoned_ids
    }
    torch.save(backdoor_data,save_path)
    print(f"backdoor_data is saved in {save_path}")
