from torch.utils.data import Dataset
from torch.utils.data import Subset, ConcatDataset
from torch.utils.data import DataLoader
from collections import Counter

class ExtractDataset(Dataset):
    '''
    抽取数据集到一个list中,目的是加快速度
    '''
    def __init__(self, old_dataset):
        self.old_dataset = old_dataset
        self.new_dataset = self._get_new_dataset()

    def _get_new_dataset(self):
        # 将数据集加载到内存中了
        new_dataset = []
        for id in range(len(self.old_dataset)):
            sample,label,isPoisoned = self.old_dataset[id]
            new_dataset.append((sample,label,isPoisoned))
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label,isPoisoned=self.new_dataset[index]
        return sample,label,isPoisoned

class ExtractDataset_NoPoisonedFlag(Dataset):
    '''
    抽取数据集到一个list中,目的是加快速度
    '''
    def __init__(self, old_dataset):
        self.old_dataset = old_dataset
        self.new_dataset = self._get_new_dataset()

    def _get_new_dataset(self):
        # 将数据集加载到内存中了
        new_dataset = []
        for id in range(len(self.old_dataset)):
            sample,label = self.old_dataset[id]
            new_dataset.append((sample,label))
        return new_dataset
    
    def __len__(self):
        return len(self.new_dataset)
    
    def __getitem__(self, index):
        sample,label=self.new_dataset[index]
        return sample,label

def split_dataset(clean_trainset, selected_indices):
    all_indices = set(range(len(clean_trainset)))
    indices_to_keep = list(all_indices-set(selected_indices))
    origin_subset = Subset(clean_trainset, indices_to_keep)
    to_adv_subset = Subset(clean_trainset, selected_indices)
    return origin_subset,to_adv_subset

def check_labels(dataset):
    train_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
    labels = []
    for batch in train_loader:
        y = batch[1]
        labels.extend(y.tolist())
    print(Counter(labels))