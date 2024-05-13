import copy
import numpy as np
from torch.utils.data.dataset import Dataset

class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
        labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
            set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)
        self.labeled = labeled
        # self.prefetch = self.dataset.prefetch
        # if self.prefetch:
        #     self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if self.labeled:            
            item = self.dataset[self.semi_indice[index]]
            sample = item[0]
            label = item[1]
            labeled = True
            itme 
        else:
            item_1 = self.dataset[self.semi_indice[index]]
            item_2 = self.dataset[self.semi_indice[index]]
            img1, img2 = item_1[0], item_2[0]
            item1.update({"img1": img1, "img2": img2})
            item = item1
            item["labeled"] = False

        return item

    def __len__(self):
        return len(self.semi_indice)