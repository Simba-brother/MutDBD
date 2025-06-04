# ASD baseline mian file

import joblib
import setproctitle
import os
import time
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader # 用于批量加载训练集的


from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop
from torchvision.datasets import DatasetFolder
from codes import config
from codes.core.attacks import BadNets
from codes.core.models.resnet import ResNet
from codes.asd.loss import SCELoss, MixMatchLoss
from codes.asd.semi import poison_linear_record, mixmatch_train,linear_test
from codes.asd.dataset import MixMatchDataset
# from ASD.log import result2csv
from codes.utils import create_dir
from codes.scripts.dataset_constructor import ExtractDataset, PureCleanTrainDataset, PurePoisonedTrainDataset
# from ASD.models.resnet_cifar import get_model

def main_test():
    # 进程名称
    proctitle = f"ASD|{config.dataset_name}|{config.model_name}|{config.attack_name}"
    setproctitle.setproctitle(proctitle)
    print(f"proctitle:{proctitle}")
    # 准备数据集
    dataset = prepare_data()
    trainset = dataset["trainset"]
    testset = dataset["testset"]
    # 准备victim model
    model = prepare_model()
    # 攻击
    backdoor_data = backdoor_attack(trainset,testset,model)
    # ASD防御训练
    defence_train(
        model = prepare_model(),
        class_num = config.class_num,
        poisoned_train_dataset = backdoor_data["poisoned_train_dataset"], # 有污染的训练集
        poisoned_ids = backdoor_data["poisoned_ids"], # 被污染的样本id list
        poisoned_eval_dataset_loader = backdoor_data["poisoned_eval_dataset_loader"], # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader = backdoor_data["poisoned_train_dataset_loader"], #有污染的训练集加载器
        clean_test_dataset_loader = backdoor_data["clean_test_dataset_loader"], # 干净的测试集加载器
        poisoned_test_dataset_loader = backdoor_data["poisoned_test_dataset_loader"], # 污染的测试集加载器
        device = torch.device(f"cuda:{config.gpu_id}"),
        save_dir = os.path.join(config.exp_root_dir, "ASD", config.dataset_name, config.model_name, config.attack_name)
    )
    
def backdoor_attack(trainset, testset, model, random_seed=666, deterministic=True):
    '''
    攻击方法：
    Args:
        trainset: 训练集(已经经过了普通的transforms)
        trainset: 测试集(已经经过了普通的transforms)
        model: victim model
    Return:
        攻击后的字典数据
    '''
    if config.attack_name == "BadNets":
        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[-3:, -3:] = 255
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-3:, -3:] = 1.0
              
        # torch.manual_seed(global_seed)
        # np.random.seed(global_seed)
        # random.seed(global_seed)
        badnets = BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=model,
            loss=nn.CrossEntropyLoss(),
            y_target=config.target_class_idx,
            poisoned_rate=0.1,
            pattern=pattern,
            weight=weight,
            seed=random_seed,
            poisoned_transform_train_index= -1,
            poisoned_transform_test_index= -1,
            poisoned_target_transform_index=0,
            deterministic=deterministic
        )
        # 被污染的训练集
        poisoned_train_dataset = badnets.poisoned_train_dataset
        # 数据集中被污染的实例id
        poisoned_ids = poisoned_train_dataset.poisoned_set
        # 被污染的训练集加载器
        poisoned_train_dataset_loader = DataLoader(
            poisoned_train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            )
        # 验证集加载器（可以是被污染的训练集但是不打乱）
        poisoned_eval_dataset_loader = DataLoader(
            poisoned_train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
        # 被污染的测试集
        poisoned_test_dataset = badnets.poisoned_test_dataset
        # 被污染的测试集加载器
        poisoned_test_dataset_loader = DataLoader(
            poisoned_test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
        # 干净污染的测试集加载器
        clean_test_dataset_loader = DataLoader(
            testset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            )
        exp_root_dir = "/data/mml/backdoor_detect/experiments"
        dataset_name = "CIFAR10"
        model_name = "ResNet18"
        attack_name = "BadNets"
        schedule = {
            'device': f'cuda:{config.gpu_id}',
            'benign_training': False,
            'batch_size': 128,
            'num_workers': 4,

            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'schedule': [150, 180], # epoch区间

            'epochs': 200,

            'log_iteration_interval': 100,
            'test_epoch_interval': 10,
            'save_epoch_interval': 10,

            'save_dir': os.path.join(exp_root_dir, "attack", dataset_name, model_name, attack_name),
            'experiment_name': 'attack'
        }
        badnets.train(schedule)
        # 工作dir
        work_dir = badnets.work_dir
        # 获得backdoor model weights
        backdoor_model = badnets.best_model
        # clean testset
        clean_testset = testset
        # poisoned testset
        poisoned_testset = badnets.poisoned_test_dataset
        # poisoned trainset
        poisoned_trainset = badnets.poisoned_train_dataset
        # poisoned_ids
        poisoned_ids = poisoned_trainset.poisoned_set
        # pure clean trainset
        pureCleanTrainDataset = PureCleanTrainDataset(poisoned_trainset, poisoned_ids)
        # pure poisoned trainset
        purePoisonedTrainDataset = PurePoisonedTrainDataset(poisoned_trainset, poisoned_ids)
        dict_state = {}
        dict_state["backdoor_model"] = backdoor_model
        dict_state["poisoned_trainset"]=poisoned_trainset
        dict_state["poisoned_ids"]=poisoned_ids
        dict_state["pureCleanTrainDataset"] = pureCleanTrainDataset
        dict_state["purePoisonedTrainDataset"] = purePoisonedTrainDataset
        dict_state["clean_testset"]=clean_testset
        dict_state["poisoned_testset"]=poisoned_testset
        dict_state["pattern"] = pattern
        dict_state['weight']=weight
        save_file_name = f"dict_state.pth"
        save_path = os.path.join(work_dir, save_file_name)
        torch.save(dict_state, save_path)
        print(f"BadNets攻击完成,数据和日志被存入{save_path}")
    res = {
        "backdoor_model":badnets.best_model,
        "poisoned_train_dataset":poisoned_train_dataset,
        "poisoned_ids":poisoned_ids,
        "poisoned_train_dataset_loader":poisoned_train_dataset_loader,
        "poisoned_eval_dataset_loader":poisoned_eval_dataset_loader,
        "poisoned_test_dataset":poisoned_test_dataset,
        "poisoned_test_dataset_loader":poisoned_test_dataset_loader,
        "clean_testset":testset,
        "clean_trainset":trainset,
        "clean_test_dataset_loader":clean_test_dataset_loader,
    }
    return res
    
def prepare_model():
    '''
    准备模型
    Return:
        model
    '''
    if config.model_name == "ResNet18":
        model = ResNet(num=18, num_classes=config.class_num)
    return model

def prepare_data():
    '''
    准备训练集和测试集
    '''
    # 加载数据集
    # 训练集transform    
    transform_train = Compose([
        # Convert a tensor or an ndarray to PIL Image
        ToPILImage(), 
        # 训练数据增强,随机水平翻转 
        RandomCrop(size=32,padding=4,padding_mode="reflect"), # img (PIL Image or Tensor): Image to be cropped.
        RandomHorizontalFlip(),
        ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    ])
    # 测试集transform
    transform_test = Compose([
        ToPILImage(),
        ToTensor()
    ])
    # 数据集文件夹
    dataset_dir = config.CIFAR10_dataset_dir
    # 获得训练数据集
    trainset = DatasetFolder(
        root=os.path.join(dataset_dir, "train"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    # 获得测试数据集
    testset = DatasetFolder(
        root=os.path.join(dataset_dir, "test"),
        loader=cv2.imread, # ndarray (H,W,C)
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    res = {
        "trainset":trainset,
        "testset":testset,
    }
    return res

def defence_train(
        model, # victim model
        class_num, # 分类数量
        poisoned_train_dataset, # 预制的有污染的训练集
        poisoned_ids, # 被污染的样本id list
        poisoned_eval_dataset_loader, # 有污染的验证集加载器（可以是有污染的训练集不打乱加载）
        poisoned_train_dataset_loader, #有污染的训练集加载器,打乱顺序加载
        clean_test_dataset_loader, # 干净的测试集加载器
        poisoned_test_dataset_loader, # 污染的测试集加载器
        device, # GPU设备对象
        save_dir, # 实验结果存储目录
        logger=None,
        **kwargs
        ):
    '''
    ASD防御训练方法
    '''
    np.random.seed(kwargs["rand_seed"])
    # 模型放gpu上
    model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 损失函数对象放到gpu上
    criterion.to(device)
    # 用于分割的损失函数
    split_criterion = SCELoss(alpha=0.1, beta=1, num_classes=class_num)
    # 分割损失函数对象放到gpu上
    split_criterion.to(device)
    # semi 损失函数
    semi_criterion = MixMatchLoss(rampup_length=config.asd_config[kwargs["dataset_name"]]["epoch"], lambda_u=15) # rampup_length = 120  same as epoches
    # 损失函数对象放到gpu上
    semi_criterion.to(device)
    # 模型参数的优化器
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # 先选择clean seed
    clean_data_info = {} # {cls_id:[sample_id],}
    all_data_info = {} # {cls_id:[sample_id],}
    for class_idx in range(class_num):
        clean_data_info[class_idx] = []
        all_data_info[class_idx] = []
    # 遍历poisoned_trainset(新鲜的)
    for idx, item in enumerate(poisoned_train_dataset):
        sample = item[0]
        label = item[1]
        if idx not in poisoned_ids:
            clean_data_info[label].append(idx)
        all_data_info[label].append(idx)
    # 选出的clean seed idx，每个类别选10个样本
    choice_clean_indice = []
    for class_idx, idx_list in clean_data_info.items():
        # 从每个class_idx中选择10个sample idx
        choice_list = np.random.choice(idx_list, replace=False, size=10).tolist()
        choice_clean_indice.extend(choice_list)
        # 从all_data_info中剔除选择出的clean seed sample index
        all_data_info[class_idx] = [x for x in all_data_info[class_idx] if x not in choice_list]
    # 存储所有选择的种子样本id
    choice_clean_indice = np.array(choice_clean_indice)

    choice_num = 0 # 每个轮次选择的数量
    best_acc = -1 # 干净测试集上的最好性能
    best_epoch = -1 # 记录下最好的epoch_id
    # 从配置文件中读取总共的训练轮次
    total_epoch = config.asd_config[kwargs["dataset_name"]]["epoch"]
    for epoch in range(total_epoch):
        epoch_start_time = time.perf_counter()
        logger.info("===Epoch: {}/{}===".format(epoch+1, total_epoch))
        if epoch < 60: # [0,59]
            # 记录下样本的loss,feature,label,方便进行clean数据的挖掘
            # 对全体数据集进行评估（耗时）
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device, dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"] )
            if epoch % 5 == 0 and epoch != 0:
                # 每五个epoch 每个class中选择数量就多加10个
                choice_num += 10
            logger.info("Mining clean data by class-aware loss-guided split...")
            # 0表示在污染池,1表示在clean pool
            split_indice = class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, 
                                                         choice_num, poisoned_ids,logger)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < 90: # [60,89]
            # 使用此时训练状态的model对数据集进行record(记录下样本的loss,feature,label,方便进行clean数据的挖掘)
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device,dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])
            logger.info("Mining clean data by class-agnostic loss-guided split...")
            split_indice = class_agnostic_loss_guided_split(record_list, 0.5, poisoned_ids,logger)
            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)
        elif epoch < total_epoch:
            # 使用此时训练状态的model对数据集进行record(记录下样本的loss,feature,label,方便进行clean数据的挖掘)
            record_list = poison_linear_record(model, poisoned_eval_dataset_loader, split_criterion, device,dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])
            meta_virtual_model = deepcopy(model)
            dataset_name = kwargs["dataset_name"]
            model_name = kwargs["model_name"]
            if dataset_name in ["CIFAR10","GTSRB"]:
                if model_name == "ResNet18":
                    # 元虚拟模型要更新的参数
                    param_meta = [  
                                    {'params': meta_virtual_model.layer3.parameters()},
                                    {'params': meta_virtual_model.layer4.parameters()},
                                    # {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
                elif model_name == "VGG19":
                    param_meta = [  
                                    {'params': meta_virtual_model.classifier_1.parameters()},
                                    {'params': meta_virtual_model.classifier_2.parameters()},
                                ]
                elif model_name == "DenseNet":
                    param_meta = [  
                                    {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.dense4.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
            elif dataset_name == "ImageNet2012_subset":
                if model_name == "ResNet18":
                    # 元虚拟模型要更新的参数
                    param_meta = [  
                                    {'params': meta_virtual_model.layer3.parameters()},
                                    {'params': meta_virtual_model.layer4.parameters()},
                                    # {'params': meta_virtual_model.linear.parameters()},
                                    {'params': meta_virtual_model.fc.parameters()}
                                ]
                elif model_name == "VGG19":
                    param_meta = [  
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
                elif model_name == "DenseNet":
                    param_meta = [  
                                    {'params': meta_virtual_model.features.denseblock4.parameters()},
                                    {'params': meta_virtual_model.classifier.parameters()}
                                ]
            # 元模型的参数优化器
            meta_optimizer = torch.optim.Adam(param_meta, lr=0.015)
            # 元模型的损失函数
            meta_criterion = nn.CrossEntropyLoss(reduction="mean")
            meta_criterion.to(device)
            for _ in range(1):
                # 使用完整的训练集训练一轮元模型
                train_the_virtual_model(
                                        meta_virtual_model=meta_virtual_model, 
                                        poison_train_loader = poisoned_train_dataset_loader,
                                        meta_optimizer=meta_optimizer,
                                        meta_criterion=meta_criterion,
                                        device = device
                                        )
            # 使用元模型对数据集进行一下record      
            meta_record_list = poison_linear_record(meta_virtual_model, poisoned_eval_dataset_loader, split_criterion, device, dataset_name=kwargs["dataset_name"], model_name =kwargs["model_name"])

            # 开始干净样本的挖掘
            logger.info("Mining clean data by meta-split...")
            split_indice = meta_split(record_list, meta_record_list, 0.5, poisoned_ids,logger)

            xdata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=True)
            udata = MixMatchDataset(poisoned_train_dataset, split_indice, labeled=False)  
        # 开始半监督训练
        # 开始clean pool进行监督学习,poisoned pool进行半监督学习
        batch_size = 64
        xloader = DataLoader(xdata, batch_size=64, num_workers=16, pin_memory=True, shuffle=True, drop_last=True)
        uloader = DataLoader(udata, batch_size=64, num_workers=16, pin_memory=True, shuffle=True, drop_last=True)
        logger.info("MixMatch training...")
        # 半监督训练参数,固定1024个batch
        semi_mixmatch = {"train_iteration": 1024,"temperature": 0.5, "alpha": 0.75,"num_classes": class_num}
        poison_train_result = mixmatch_train(
            model,
            xloader,
            uloader,
            semi_criterion,
            optimizer,
            epoch,
            device,
            logger,
            **semi_mixmatch
        )
        epoch_end_time = time.perf_counter()
        epcoch_cost_time = epoch_end_time - epoch_start_time
        hours = int(epcoch_cost_time // 3600)
        minutes = int((epcoch_cost_time % 3600) // 60)
        seconds = epcoch_cost_time % 6
        logger.info(f"Epoch耗时:{hours}时{minutes}分{seconds:.3f}秒")
        logger.info("Test model on clean data...")
        clean_test_result = linear_test(
            model, clean_test_dataset_loader, criterion,device,logger
        )

        logger.info("Test model on poison data...")
        poison_test_result = linear_test(
            model, poisoned_test_dataset_loader, criterion,device,logger
        )

        # if scheduler is not None:
        #     scheduler.step()
        #     logger.info(
        #         "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
        #     )

        # Save result and checkpoint.
        # 保存结果
        result = {
            "poison_train": poison_train_result, # 训练集上结果
            "clean_test": clean_test_result, # 干净测试集上结果
            "poison_test": poison_test_result, # 中毒测试集上结果
        }
        
        result_epochs_dir = os.path.join(save_dir, "result_epochs")
        create_dir(result_epochs_dir)
        save_file_name = f"result_epoch_{epoch}.data"
        save_file_path = os.path.join(result_epochs_dir, save_file_name)
        joblib.dump(result,save_file_path)
        logger.info(f"epoch:{epoch},result: is saved in {save_file_path}")

        
        # result2csv(result, save_dir)
       
        # if scheduler is not None:
        #     saved_dict["scheduler_state_dict"] = scheduler.state_dict()

        is_best = False
        if clean_test_result["acc"] > best_acc:
            # 干净集测试acc的best
            is_best = True
            best_acc = clean_test_result["acc"]
            best_epoch = epoch
         # 保存状态
         # 每个训练轮次的状态
        saved_dict = {
            "epoch": epoch,
            "result": result,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc, # clean testset上的acc
            "best_epoch": best_epoch,
        }
        logger.info("Best test accuaracy {} in epoch {}".format(best_acc, best_epoch))
        # 每当best acc更新后，保存checkpoint
        ckpt_dir = os.path.join(save_dir, "ckpt")
        create_dir(ckpt_dir)
        if is_best:
            # clean testset acc的best model
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(saved_dict, best_ckpt_path)
            print("Save the best model to {}".format(best_ckpt_path))
        # 保存最新一次checkpoint
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_model.pt")
        torch.save(saved_dict, latest_ckpt_path)
        logger.info("Save the latest model to {}".format(latest_ckpt_path))
        # 保存倒数第2轮次的模型权重
        if epoch == total_epoch-2:
            secondtolast_path = os.path.join(ckpt_dir,"secondtolast.pth")
            torch.save(model.state_dict(),secondtolast_path)
            print("Save the secondtolast model to {}".format(secondtolast_path))
        # 保存第第59轮次结束的模型权重
        if epoch == 59:
            epoch_59_path = os.path.join(ckpt_dir,"epoch_59.pth")
            torch.save(model.state_dict(),epoch_59_path)
            logger.info("Save the secondtolast model to {}".format(epoch_59_path))
    
    logger.info("ASD_train_End")
    return best_ckpt_path,latest_ckpt_path

def class_aware_loss_guided_split(record_list, choice_clean_indice, all_data_info, choice_num, poisoned_indice, logger):
    """
    Adaptively split the poisoned dataset by class-aware loss-guided split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    clean_pool_flag = np.zeros(len(loss)) # 每次选择都清空
    # 存总共选择的clean 的idx,包括seed和loss最低的的sample idx
    total_indice = choice_clean_indice.tolist() # choice_clean_indice装的seed
    for class_idx, sample_indice in all_data_info.items():
        # 遍历每个class_idx
        sample_indice = np.array(sample_indice)
        loss_class = loss[sample_indice]
        # 选择SCE loss较低的
        indice_class = loss_class.argsort()[: choice_num]
        indice = sample_indice[indice_class]
        # list的extend操作
        total_indice += indice.tolist()
    # 统计构建出的clean pool 其中可能还混有污染样本的数量,这里我们统计一下
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count+=1
    total_indice = np.array(total_indice)
    clean_pool_flag[total_indice] = 1 # 1表示clean

    logger.info(
        "{}/{} poisoned samples in clean data pool".format(poisoned_count, clean_pool_flag.sum())
    )
    return clean_pool_flag

def class_agnostic_loss_guided_split(record_list, ratio, poisoned_indice,logger):
    """
    Adaptively split the poisoned dataset by class-agnostic loss-guided split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    clean_pool_flag = np.zeros(len(loss))
    total_indice = loss.argsort()[: int(len(loss) * ratio)]
    # 统计构建出的clean pool 中还混有污染样本的数量
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count+=1
    logger.info(
        "{}/{} poisoned samples in clean data pool".format(poisoned_count, len(total_indice))
    )
    clean_pool_flag[total_indice] = 1
    return clean_pool_flag

def meta_split(record_list, meta_record_list, ratio, poisoned_indice,logger):
    """
    Adaptively split the poisoned dataset by meta-split.
    """
    keys = [record.name for record in record_list]
    loss = record_list[keys.index("loss")].data.numpy()
    meta_loss = meta_record_list[keys.index("loss")].data.numpy()
    clean_pool_flag = np.zeros(len(loss))
    loss = loss - meta_loss
    # dif小的样本被选择为clean样本
    total_indice = loss.argsort()[: int(len(loss) * ratio)]
    poisoned_count = 0
    for idx in total_indice:
        if idx in poisoned_indice:
            poisoned_count += 1
    logger.info("{}/{} poisoned samples in clean data pool".format(poisoned_count, len(total_indice)))
    clean_pool_flag[total_indice] = 1
    predict_p_idx_list = loss.argsort()[int(len(loss) * ratio):]
    tp_num= len(set(predict_p_idx_list) & set(poisoned_indice))
    recall = round(tp_num/len(poisoned_indice),4)
    precision = round(tp_num / len(predict_p_idx_list),4)
    f1 = 2*recall*precision/(precision+recall+1e-10)
    logger.info(f"recall:{recall},precison:{precision},f1:{f1}")
    return clean_pool_flag

def train_the_virtual_model(meta_virtual_model, poison_train_loader, meta_optimizer, meta_criterion, device):
    """
    Train the virtual model in meta-split.
    """
    meta_virtual_model.train()
    for batch_idx, batch in enumerate(poison_train_loader):
        data = batch[0]
        target = batch[1]
        data = data.to(device)
        target = target.to(device)
        # 优化器中的参数梯度清零
        meta_optimizer.zero_grad()
        output = meta_virtual_model(data)
        meta_criterion.reduction = "mean"
        # 损失函数
        loss = meta_criterion(output, target)
        # 损失函数对虚拟模型参数求导
        loss.backward()
        # 优化器中的参数更新
        meta_optimizer.step()

if __name__ == "__main__":
    main_test()