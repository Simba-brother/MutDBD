from torch.utils.data import DataLoader
def get_data_loder(dataset,shuffle,
                   batch_size:int = 128,
                   num_workers:int = 8,
                   drop_last:bool = False,
                   pin_memory:bool = True
                   ):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
            )