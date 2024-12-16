'''
灵活构建数据集脚本
'''
import torch
import copy
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.datasets import DatasetFolder

class PureCleanTrainDataset(Dataset):
    '''
    构建出干净的训练集
    '''
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        # 含有污染的训练集
        self.poisoned_train_dataset = poisoned_train_dataset
        # 被污染的ids
        self.poisoned_ids  = poisoned_ids
        # 干净的训练集（type:list）
        self.pureCleanTrainDataset = self._getPureCleanTrainDataset()

    def _getPureCleanTrainDataset(self):
        pureCleanTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label, isPoisoned = self.poisoned_train_dataset[id][0], self.poisoned_train_dataset[id][1], self.poisoned_train_dataset[id][2]
            if id not in self.poisoned_ids:
                pureCleanTrainDataset.append((sample,label,isPoisoned))
        return pureCleanTrainDataset
    
    def __len__(self):
        # 训练集的长度
        return len(self.pureCleanTrainDataset)
    
    def __getitem__(self, index):
        # 根据索引检索样本
        sample,label,isPoisoned=self.pureCleanTrainDataset[index]
        return sample,label,isPoisoned

class PurePoisonedTrainDataset(Dataset):
    '''
    构建出纯污染的训练集
    '''
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.purePoisonedTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        purePoisonedTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label, isPoisoned = self.poisoned_train_dataset[id][0], self.poisoned_train_dataset[id][1], self.poisoned_train_dataset[id][2]
            if id in self.poisoned_ids:
                purePoisonedTrainDataset.append((sample,label,isPoisoned))
        return purePoisonedTrainDataset
    
    def __len__(self):
        return len(self.purePoisonedTrainDataset)
    
    def __getitem__(self, index):
        sample,label,isPoisoned=self.purePoisonedTrainDataset[index]
        return sample,label,isPoisoned

class ExtractDataset(Dataset):
    '''
    抽取数据集到一个list中,目的是加快速度
    '''
    def __init__(self, old_dataset):
        self.old_dataset = old_dataset
        self.new_dataset = self._get_new_dataset()

    def _get_new_dataset(self):
        new_dataset = []
        for id in range(len(self.old_dataset)):
            sample, label, isPoisoned = self.old_dataset[id][0], self.old_dataset[id][1], self.old_dataset[id][2]
            new_dataset.append((sample,label,isPoisoned))
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label,isPoisoned=self.new_dataset[index]
        return sample,label,isPoisoned
    
class IAD_Dataset(Dataset):
    '''
    构建出IAD的数据集
    '''
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.dataset = self._get_dataset()

    def _get_dataset(self):
        dataset = []
        for id in range(len(self.data)):
            sample, label =  torch.tensor(self.data[id]), torch.tensor(self.label[id])
            dataset.append((sample, label))
        assert len(dataset) == len(self.data), "数量不对"
        return dataset 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x,y = self.dataset[index][0], self.dataset[index][1]
        return x,y
    
class ExtractTargetClassDataset(Dataset):
    '''
    从数据集中抽取出某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx = target_class_idx
        self.targetClassDataset = self._getTargetClassDataset()

    def _getTargetClassDataset(self):
        targetClassDataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id][0], self.dataset[id][1]
            if label == self.target_class_idx:
                targetClassDataset.append((sample,label))
        return targetClassDataset
    
    def __len__(self):
        return len(self.targetClassDataset)
    
    def __getitem__(self, index):
        sample,label =self.targetClassDataset[index]
        return sample,label

class ExtractCleanTargetClassDataset(Dataset):
    '''
    从数据集中抽取出某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset, target_class_idx, poisoned_ids):
        self.dataset = dataset
        self.target_class_idx = target_class_idx
        self.poisoned_ids = poisoned_ids
        self.targetCleanClassDataset = self._getCleanTargetClassDataset()

    def _getCleanTargetClassDataset(self):
        targetCleanClassDataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id][0], self.dataset[id][1]
            if label == self.target_class_idx and id not in self.poisoned_ids:
                targetCleanClassDataset.append((sample,label))
        return targetCleanClassDataset
    
    def __len__(self):
        return len(self.targetCleanClassDataset)
    
    def __getitem__(self, index):
        sample,label =self.targetCleanClassDataset[index]
        return sample,label
    
class ExtractPoisonedTargetClassDataset(Dataset):
    '''
    从数据集中抽取出某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset, target_class_idx, poisoned_ids):
        self.dataset = dataset
        self.target_class_idx = target_class_idx
        self.poisoned_ids = poisoned_ids
        self.targetPoisonedClassDataset = self._getPoisonedTargetClassDataset()

    def _getPoisonedTargetClassDataset(self):
        targetPoisonedClassDataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id][0], self.dataset[id][1]
            if label == self.target_class_idx and id in self.poisoned_ids:
                targetPoisonedClassDataset.append((sample,label))
        return targetPoisonedClassDataset
    
    def __len__(self):
        return len(self.targetPoisonedClassDataset)
    
    def __getitem__(self, index):
        sample,label =self.targetPoisonedClassDataset[index]
        return sample,label

class CombinDataset(Dataset):
    '''
    两个数据集进行合并
    '''
    def __init__(self, dataset_1, dataset_2):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.new_dataset = self._get_combin_dataset()

    def _get_combin_dataset(self):
        new_dataset = []
        for i in range(len(self.dataset_1)):
            new_dataset.append(self.dataset_1[i]) 
        for i in range(len(self.dataset_2)):
            new_dataset.append(self.dataset_2[i])
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label =self.new_dataset[index]
        return sample,label
    
class ExtractDatasetByIds(Dataset):
    '''
    从数据集中抽取特定ids的子集
    '''
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids
        self.new_dataset = self._get_dataset_by_ids()


    def _get_dataset_by_ids(self):
        new_dataset = []
        for id in self.ids:
            new_dataset.append(self.dataset[id])
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label =self.new_dataset[index]
        return sample,label

class ExtractDatasetAndModifyLabel(Dataset):
    '''
    从数据集中抽取特定targets的子集
    '''
    def __init__(self, dataset, label_remapp):
        self.dataset = dataset
        self.label_remapp = label_remapp
        self.new_dataset = self._get_new_dataset()


    def _get_new_dataset(self):
        new_dataset = []
        for i in range(len(self.dataset)):
            sample, label = self.dataset[i]
            new_label = self.label_remapp[label]
            new_dataset.append((sample, new_label))
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label =self.new_dataset[index]
        return sample,label

class ExtractNoTargetClassDataset(Dataset):
    '''
    从数据集中抽取出不含某个类别(target_class_idx)的数据集
    '''
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx = target_class_idx
        self.no_targetClassDataset = self._getTargetClassDataset()

    def _getTargetClassDataset(self):
        no_targetClassDataset = []
        for id in range(len(self.dataset)):
            sample, label = self.dataset[id][0], self.dataset[id][1]
            if label != self.target_class_idx:
                no_targetClassDataset.append((sample,label))
        return no_targetClassDataset
    
    def __len__(self):
        return len(self.no_targetClassDataset)
    
    def __getitem__(self, index):
        sample,label =self.no_targetClassDataset[index]
        return sample,label

class Add_IAD_DatasetFolderTrigger():
    """Add IAD trigger to DatasetFolder images.
    """

    def __init__(self, modelG, modelM):
         self.modelG = modelG
         self.modelM = modelM

    def __call__(self, img):
        # 允许一个类的实例像函数一样被调用
        """Get the poisoned image..
        img: shap:CHW,type:Tensor
        """
        
        # 添加一个维度索引构成BCHW
        imgs = img.unsqueeze(0) # 增加一个B维度
        # G model生成pattern
        patterns = self.modelG(imgs)
        # 对pattern normalize一下
        patterns = self.modelG.normalize_pattern(patterns)
        # 获得masks
        masks_output = self.modelM.threshold(self.modelM(imgs))
        # inputs, patterns, masks => bd_inputs
        bd_imgs = imgs + (patterns - imgs) * masks_output 
        # 压缩一个维度
        bd_img = bd_imgs.squeeze(0) # 不会replace
        bd_img= bd_img.detach()
        return bd_img

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class IADPoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 modelG,
                 modelM
                 ):
        super(IADPoisonedDatasetFolder, self).__init__(
            # 数据集文件夹位置
            benign_dataset.root, # 数据集文件夹 /data/mml/backdoor_detect/dataset/cifar10/train
            # 数据集直接加载器
            benign_dataset.loader, # cv2.imread
            # 数据集扩展名
            benign_dataset.extensions, # .png
            # 数据集transform
            benign_dataset.transform, # 被注入到self.transform
            # 数据集标签transform
            benign_dataset.target_transform, # 对label进行transform
            None)
        # 数据集包含的数据量
        total_num = len(benign_dataset)
        # 需要中毒的数据量
        poisoned_num = int(total_num * poisoned_rate)
        # 断言：中毒的数据量必须是>=0
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        # 数据id list
        tmp_list = list(range(total_num)) #[0,1,2,...,N]
        # id list被打乱
        random.shuffle(tmp_list)
        # 选出的id set作为污染目标样本
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        # Add trigger to images
        # 注意在调用父类（DatasetFolder）构造时self.transform = benign_dataset.transform
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform) # Compose()的深度拷贝    
        # 中毒转化器为在普通样本转化器前再加一个AddDatasetFolderTrigger
        self.poisoned_transform.transforms.append(Add_IAD_DatasetFolderTrigger(modelG, modelM))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.append(ModifyTarget(y_target))

    def __getitem__(self, index):
        # DatasetFolder 必须要有迭代
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 父类（DatasetFolder）属性
        sample = self.loader(path) # self.loader也是调用父类构造时注入的
        isPoisoned = False
        if index in self.poisoned_set: # self.poisoned_set 本类构造注入的
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
            isPoisoned = True
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target, isPoisoned

