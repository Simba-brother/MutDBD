from torch.utils.data import Dataset, dataloader
import torch


class PureCleanTrainDataset(Dataset):
    def __init__(self, poisoned_train_dataset, poisoned_ids):
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_ids  = poisoned_ids
        self.pureCleanTrainDataset = self._getPureCleanTrainDataset()
    def _getPureCleanTrainDataset(self):
        pureCleanTrainDataset = []
        for id in range(len(self.poisoned_train_dataset)):
            sample, label, isPoisoned = self.poisoned_train_dataset[id][0], self.poisoned_train_dataset[id][1], self.poisoned_train_dataset[id][2]
            if id not in self.poisoned_ids:
                pureCleanTrainDataset.append((sample,label,isPoisoned))
        return pureCleanTrainDataset
    
    def __len__(self):
        return len(self.pureCleanTrainDataset)
    
    def __getitem__(self, index):
        sample,label,isPoisoned=self.pureCleanTrainDataset[index]
        return sample,label,isPoisoned

class PurePoisonedTrainDataset(Dataset):
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



