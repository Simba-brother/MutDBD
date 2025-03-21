import os
import random


import wandb
os.environ["WANDB_API_KEY"] = '84370b23b213e98b40ce4df7e8e6b93f4374978d'
os.environ["WANDB_MODE"] = "offline"



# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="test-project",

    # 跟踪超参数并运行元数据
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 15
offset = random.random() / 5
for epoch in range(2, epochs):
    # 每个epoch的acc
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    # 每个epoch的loss
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
'''
import numpy as np
from PIL import Image
# 储存运行过程中的图像：随机生成一个图像作为示例
data = np.random.rand(256, 256, 3) * 255
data = data.astype(np.uint8)
image = Image.fromarray(data, 'RGB')

# 储存运行过程中的loss等日志
wandb.log(
    {
        "epoch": epoch,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        'images': wandb.Image(image),
    }
)

wandb.run.summary['Trainable parameters'] = f"{n_params / 1e6}M"

wandb.log(
    {"Model_architecture": wandb.Table(columns=["Model_architecture"], data=[[str(model_without_ddp)]])}
)

wandb.init(project="config_example", id="ismjl25l", resume="must")

import wandb
import argparse
import numpy as np
import random
from PIL import Image
 
# 初始化wandb
wandb.init(project="config_example")
 
 
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss
 
 
def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss
 
 
def main(args):
    # 储存运行参数：将参数值转为dict，然后再储存
    wandb.config.update(vars(args))
 
    for epoch in np.arange(1, args.epochs):
        train_acc, train_loss = train_one_epoch(epoch, args.learning_rate, args.batch_size)
        val_acc, val_loss = evaluate_one_epoch(epoch)
 
        # 储存运行过程中的图像：随机生成一个图像作为示例
        data = np.random.rand(256, 256, 3) * 255
        data = data.astype(np.uint8)
        image = Image.fromarray(data, 'RGB')
 
        # 储存运行过程中的loss等日志
        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                'images': wandb.Image(image),
            }
        )
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="Learning rate")
 
    args = parser.parse_args()
    main(args)
'''