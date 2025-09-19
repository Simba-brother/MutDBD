from torch.utils.data import Dataset
class PoisonedDataset(Dataset):
    def __init__(self,dataset,poisoned_set,target_class,train_or_teset):
        self.dataset = dataset
        self.poisoned_set = poisoned_set
        self.target_class = target_class
        self.train_or_teset = train_or_teset
    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if idx in self.poisoned_set:
            k = 5
            img[:,:k,:k] = 1.0
            img[:,:k,-k:] = 1.0
            img[:,-k:,:k] = 1.0
            img[:,-k:,-k:] = 1.0
            if self.train_or_teset == "test":
                label = self.target_class
        return img,label

    