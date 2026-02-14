### Code Description:

**Backdoor model performance evaluation code (ASR, ACC):** backdoor_eval.py

**Mutation model generation:** defense/our/mutation/model_mutation.py

**Mutation model evaluation on the poisoned training set:** defense/our/mutation/mutation_eval.py

**Class rank based on evaluation results:** defense/our/class_rank/main.py

**Defense training for our work:** defense/our/defense_train.py

**ASD defense:** defense/asd/defense_train.py

## Supplementary discussion and verification of the inconsistency between STRIP's "sample selection performance" and "final defense effect".

| CIFAR10-ResNet18 | CIFAR10-VGG19 |
|:---:|:---:|
| ![](imgs/dis_strip/CIFAR10_ResNet18_BadNets.png) | ![](imgs/dis_strip/CIFAR10_VGG19_BadNets.png) |

| CIFAR10-DenseNet | GTSRB-ResNet18 |
|:---:|:---:|
| ![](imgs/dis_strip/CIFAR10_DenseNet_BadNets.png) | ![](imgs/dis_strip/GTSRB_ResNet18_BadNets.png) |

| GTSRB-VGG19 | GTSRB-DenseNet |
|:---:|:---:|
| ![](imgs/dis_strip/GTSRB_VGG19_BadNets.png) | ![](imgs/dis_strip/GTSRB_DenseNet_BadNets.png) |

| ImageNet-ResNet18 | ImageNet-DenseNet |
|:---:|:---:|
| ![](imgs/dis_strip/ImageNet_ResNet18_BadNets.png) | ![](imgs/dis_strip/ImageNet_DenseNet_BadNets.png) |


