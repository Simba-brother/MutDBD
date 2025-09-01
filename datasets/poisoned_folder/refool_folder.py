import copy
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from attack.core.attacks.Refool import AddDatasetFolderTriggerMixin,ModifyTarget
class PoisonedDatasetFolder(DatasetFolder, AddDatasetFolderTriggerMixin):
    def __init__(self, benign_dataset, y_target, poisoned_ids, poisoned_transform_index, poisoned_target_transform_index, reflection_cadidates,\
            max_image_size=560, ghost_rate=0.49, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        """
        Args:
            reflection_cadidates (List of numpy.ndarray of shape (H, W, C) or (H, W))
            max_image_size (int): max(Height, Weight) of returned image
            ghost_rate (float): rate of ghost reflection
            alpha_b (float): the ratio of background image in blended image, alpha_b should be in $(0,1)$, set to -1 if random alpha_b is desired
            offset (tuple of 2 interger): the offset of ghost reflection in the direction of x axis and y axis, set to (0,0) if random offset is desired
            sigma (interger): the sigma of gaussian kernel, set to -1 if random sigma is desired
            ghost_alpha (interger): ghost_alpha should be in $(0,1)$, set to -1 if random ghost_alpha is desired
        """
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        # benign_dataset数量
        total_num = len(benign_dataset)
        self.poisoned_set = poisoned_ids
        
        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = transforms.Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # split transform into two pharses
        if poisoned_transform_index < 0:
            poisoned_transform_index = len(self.poisoned_transform.transforms) + poisoned_transform_index
        # 将transform分割
        self.pre_poisoned_transform = transforms.Compose(self.poisoned_transform.transforms[:poisoned_transform_index])
        self.post_poisoned_transform = transforms.Compose(self.poisoned_transform.transforms[poisoned_transform_index:])

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = transforms.Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        # 修改target
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        
        # 第二个父类的构造函数
        # Add Trigger
        AddDatasetFolderTriggerMixin.__init__(
            self, 
            total_num, # benign_dataset数量
            reflection_cadidates, 
            max_image_size, 
            ghost_rate,
            alpha_b,
            offset,
            sigma,
            ghost_alpha)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        isPoisoned = False
        if index in self.poisoned_set:
            if len(self.pre_poisoned_transform.transforms):
                # 预训练前部分先修改图像数据
                sample = self.pre_poisoned_transform(sample)
            # 加trigger
            sample = self.add_trigger(sample, index) # 第二个父类方法
            # 预训练后半部继续修改
            sample = self.post_poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
            isPoisoned = True
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target, isPoisoned
