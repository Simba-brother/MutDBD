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
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x,y



