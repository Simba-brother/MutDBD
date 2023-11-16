#### 实验对象
(Dataset, Models, Attacks, Defenses(baselines, mutations))

["cifar_10", "resnet18", "badnets", "mutation:GF"]

#### 数据
训练集污染率: 0.2 --> default:0.05
target label: 1
ASR: 0.959
clean acc: 0.901

模型权重变异率: 0.1
变异强度
    均值 0
    标准差 0.1
变异数量:10

变异模型:
ASR: 0.976
clean acc: 0.535

target label 确定：
$F(c) = \frac{1}{M}\sum_{m=1}^{M}ACC_o-ACC_m$
{0: 0.212, <font color=blue>1: 0.107</font>, 2: 0.508, 3: 0.238, 4: 0.662, 5: 0.598, 6: 0.7, 7: 0.607, 8: 0.438, 9: 0.668}

clean input entroy: 2.78
poisoned input entroy: 0.313

detect: 
    threshold: 0.4 （小于0.4的被判定为 poisoned）
    Precision: 1.0
    Recall: 0.89
    F1 score: 0.941
    

["cifar_10", "resnet18", "WaNet", "mutation:GF"]
poisoned rate: 0.1 （5000个）
backdoor model
    clean testset acc: 0.866
    ASR: 0.99
mutation method:
    GF:
        num:10
        mutate ratio: 0.2, 
        scale: 0.1
        clean train acc: 0.745 （最好这个低低！！）
        ASR: 0.986  （一定要保证这个高高！！）
        target label:
            {0: 0.276, <font color=blue>1: 0.055</font>, 2: 0.16, 3: 0.117, 4: 0.094, 5: 0.175, 6: 0.175, 7: 0.309, 8: 0.259, 9: 0.149}
        clean input entroy: 1.216
        poisoned input entroy: 0.243
        detect: 
            threshold: 0.3
            acc: 0.805
            Precision: 0.956
            Recall: 0.659
            F1 score:  0.780

["cifar_10", "resnet18", "IAD", "mutation:GF"]
poisoned rate: 0.1 （4680个）
backdoor model
    pure_train_clean_acc: 1.0 
    train_ASR: 0.993
mutation method:
    GF:
        num:10
        mutate ratio: 0.2
        scale: 0.05
        pure clean trainset acc: 0.106（最好这个低低！！）
        pure poisoned trainset ASR:  1.0 （一定要保证这个高高！！）
        