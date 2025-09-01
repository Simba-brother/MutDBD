from torch.utils.data import DataLoader

def filter_dataset(clean_test_dataset, target_label:int):
    clean_testset_label_list = []
    clean_testset_loader = DataLoader(
                clean_test_dataset,
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(clean_test_dataset)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != target_label:
            filtered_ids.append(sample_id)
    return filtered_ids