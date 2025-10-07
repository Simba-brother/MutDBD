import torch

from datasets.clean_dataset import get_clean_dataset
from models.model_loader import get_model
from utils.trainer import NeuralNetworkTrainer
from modelEvalUtils import EvalModel
from utils.dataset import get_data_loder


def main():
    # 加载数据集
    trainset, testset = get_clean_dataset(dataset_name,attack_name)
    trainset_loader = get_data_loder(trainset,shuffle=True)
    testset_loader = get_data_loder(testset,shuffle=False)
    # 加载模型
    model = get_model(dataset_name, model_name)
    device = torch.device("cuda:0")
    trainer = NeuralNetworkTrainer(model,device,init_lr = 0.001,model_dir="exp_result/model_train", experiment_name=f"{dataset_name}-{attack_name}")
    # 训练模型
    history = trainer.fit(
        train_loader=trainset_loader,
        val_loader=testset_loader,  # 这里用测试集作为验证集，实际应用中应使用单独的验证集
        epochs=50,
        early_stopping_patience=3,
        save_best=True
    )
    em = EvalModel(trainer.model,testset,device,batch_size=128)
    test_acc = em.eval_acc()
    print(f"Test acc:{test_acc}")

if __name__ == "__main__":
    # 本脚本参数
    dataset_name = "ImageNet2012_subset"
    model_name = "VGG19"
    attack_name = "BadNets"
    main()