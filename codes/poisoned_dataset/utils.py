from torch.utils.data import Subset
def filter_class(dataset, class_to_exclude):
    indices = []
    for i in range(len(dataset)):
        x,y = dataset[i]
        if y != class_to_exclude:
            indices.append(i)
    return Subset(dataset, indices)