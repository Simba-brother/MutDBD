# from torchvision.models import resnet18
import sys
sys.path.append("./")
import time
import os
import cv2
import numpy as np

import torch
import torchvision
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import pprint
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary
import timm

from codes.utils import create_dir, random_seed

def get_model():
    '''
    args:
        model_name: resnet18|xxx
        num_classes: 10 
    '''
    model = timm.create_model("resnet18", pretrained=True, num_classes = 10)
    return model

    # model = resnet18(pretrained=True)

    # for name, param in model.named_parameters():
    #     print(name,param.shape)
    # summary(model=model, input_size=(3,224,224))
    # summary(model=model, input_size=(3,224,224))
    # print(model.default_cfg)
    # print(model.get_classifier())

def get_config():
    config = {}
    # model
    # model = get_model()
    model = torch.load("experiments/CIFAR10/models/resnet18/resnet18_pretrained_224_224_3.pth")
    # transform
    transform_train=torchvision.transforms.Compose([
        # Resize step is required as we will use a ResNet model, which accepts at leats 224x224 images
        torchvision.transforms.ToPILImage(), # PIL.Image
        torchvision.transforms.Resize((224,224)),  
        torchvision.transforms.ToTensor() # C x H x W
    ])
    transform_test = Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224,224)), 
        ToTensor()
    ])
    # dataset
    trainset = DatasetFolder(
        root='./dataset/cifar10/train',
        loader=cv2.imread, # ndarray
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)

    testset = DatasetFolder(
        root='./dataset/cifar10/test',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    # dataset loader
    train_dataloader = DataLoader(
                    trainset,
                    batch_size=128,
                    shuffle=True,
                    # num_workers=self.current_schedule['num_workers'],
                    drop_last=False,
                    pin_memory=False,
                    worker_init_fn=random_seed
                )
    test_dataloader = DataLoader(
                    testset,
                    batch_size=128,
                    shuffle=False,
                    # num_workers=self.current_schedule['num_workers'],
                    drop_last=False,
                    pin_memory=False,
                    worker_init_fn=random_seed
                )
    
    lr = 1e-5
    weight_decay = 5e-4
    epochs = 20
    # Standard CrossEntropy Loss for multi-class classification problems
    criterion = torch.nn.CrossEntropyLoss()
    # params_1x are the parameters of the network body, i.e., of all layers except the FC layers
    params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=weight_decay)
    
    config["model"] = model
    config["transform_train"] = transform_train
    config["transform_test"] = transform_test
    config["trainset"] = trainset
    config["testset"] = testset
    config["train_dataloader"] = train_dataloader
    config["test_dataloader"] = test_dataloader
    config["epochs"] = epochs
    config["optimizer"] = optimizer
    config["criterion"] = criterion
    config["save_dir"] = "./experiments/CIFAR10/models/resnet18/clean"
    config["checkpoint_epochs"] = 10
    config["device"] = torch.device('cuda:2')
    return config

# 训练函数
def train(config, scheduler=None):
    model = config["model"]
    epochs = config["epochs"]
    device = config["device"]
    train_dataloader = config["train_dataloader"]
    test_dataloader = config["test_dataloader"]
    criterion = config["criterion"]
    optimizer = config["optimizer"]
    checkpoint_epochs = config["checkpoint_epochs"]
    save_dir = config["save_dir"]
    create_dir(save_dir)
    # 训练开始时间
    start = time.time()
    model.to(device)
    tb_writer = SummaryWriter(log_dir="runs/cifar10_experiment")
    # init_img = torch.zeros((1,3,224,224), device=device)
    # tb_writer.add_graph(model,init_img)
    print(f'Training for {epochs} epochs on {device}')
    best_train_acc = 0
    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        print("==="*10)
        model.train()  # put network in train mode for Dropout and Batch Normalization
        train_loss = torch.tensor(0., device=device)  # loss and accuracy tensors are on the GPU to avoid data transfers
        train_accuracy = torch.tensor(0., device=device)
        train_correct_num = 0 # 本轮次,训练集预测对的数目
        # 批次训练
        batch_id = 0
        for X, Y in train_dataloader:
            batch_id += 1
            print(f"Batch {batch_id}/{len(train_dataloader)}")
            # 训练集批次
            X = X.to(device)
            Y = Y.to(device)
            preds = model(X)
            loss = criterion(preds, Y)
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 非梯度环境
            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_correct_num += (torch.argmax(preds, dim=1) == Y).sum()
        
        if scheduler is not None: 
            scheduler.step()
        train_accuracy = train_correct_num/len(train_dataloader.dataset)
        train_loss = train_loss/len(train_dataloader.dataset)
        train_accuracy = round(train_accuracy.item(),3)
        train_loss = round(train_loss.item(),2)
        print(f'Training loss: {train_loss}')
        print(f'Training accuracy:{train_accuracy}')
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_acc", train_accuracy, epoch)
        tb_writer.add_scalar("lr",optimizer.param_groups[0]["lr"],epoch)
        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, os.path.join(save_dir,"best_model.pth"))

        if test_dataloader is not None:
            model.eval()  # put network in train mode for Dropout and Batch Normalization
            test_loss = torch.tensor(0., device=device)
            test_accuracy = torch.tensor(0., device=device)
            test_correct_num = 0
            with torch.no_grad():
                for X, Y in test_dataloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    preds = model(X)
                    loss = criterion(preds, Y)
                    test_loss += loss * test_dataloader.batch_size
                    test_correct_num += (torch.argmax(preds, dim=1) == Y).sum()
            test_accuracy = test_correct_num/len(test_dataloader.dataset)
            test_accuracy = round(test_accuracy.item(),3)
            test_loss = test_loss/len(test_dataloader.dataset)
            test_loss = round(test_loss.item(),3)
            print(f'test loss: {test_loss}')
            print(f'test accuracy: {test_accuracy}')

        save_file_name = f"epoch_{epoch}_trainAcc_{test_accuracy}_testAcc_{test_accuracy}.pth"
        second_dir = os.path.join(save_dir, "checkpoint")
        create_dir(second_dir)
        save_path = os.path.join(second_dir, save_file_name)
        if epoch%checkpoint_epochs==0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, save_path)
        print("="*10)    
    end = time.time()
    print(f'Total training time: {end-start:.1f} seconds')
    # 返回训练好的net
    return model

def process():
    # 得到config
    config = get_config()
    # 开始训练
    model = train(config)


if __name__ == "__main__":
    # process()
    get_model()


