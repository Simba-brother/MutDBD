'''
24个场景下污染数据集的生成
'''

import copy
import os
import cv2


from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose

from codes import config
from codes.transform_dataset import imagenet_Refool
from codes.core.attacks.Refool import AddDatasetFolderTriggerMixin,ModifyTarget
from codes.poisoned_dataset.utils import filter_class
from torch.utils.data import DataLoader,Subset


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
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # split transform into two pharses
        if poisoned_transform_index < 0:
            poisoned_transform_index = len(self.poisoned_transform.transforms) + poisoned_transform_index
        # 将transform分割
        self.pre_poisoned_transform = Compose(self.poisoned_transform.transforms[:poisoned_transform_index])
        self.post_poisoned_transform = Compose(self.poisoned_transform.transforms[poisoned_transform_index:])

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
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

def read_image(img_path, type=None):
    '''
    读取图片
    '''
    img = cv2.imread(img_path)
    # cv2.imshow('Image', img)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError

def gen_needed_dataset(poisoned_ids:list):

    #  数据集
    trainset,testset = imagenet_Refool()
    '''
    transform_train = Compose([
        ToPILImage(), # 在这之前投毒
        Resize((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    transform_test = Compose([
        ToPILImage(), # 在这之前投毒
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
    '''
    # backdoor pattern
    reflection_images = []
    # URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    # "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 

    # reflection image dir下所有的img path
    reflection_image_path = os.listdir(reflection_data_dir)
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    # 中毒的数据集
    # 直接在transform.Compose([])最开始处投毒
    
    poisoned_trainset = PoisonedDatasetFolder(
        trainset, 
        3, 
        poisoned_ids, 
        poisoned_transform_index=0, 
        poisoned_target_transform_index=1, 
        reflection_cadidates=reflection_images,
        max_image_size=560, ghost_rate=0.49, alpha_b=0.1, offset=(0, 0), sigma=5, ghost_alpha=-1.)
    
    # 投毒测试集
    clean_testset_label_list = []
    clean_testset_loader = DataLoader(
                testset,
                batch_size=64, 
                shuffle=False,
                num_workers=4,
                pin_memory=True)
    for _, batch in enumerate(clean_testset_loader):
        Y = batch[1]
        clean_testset_label_list.extend(Y.tolist())
    filtered_ids = []
    for sample_id in range(len(testset)):
        sample_label = clean_testset_label_list[sample_id]
        if sample_label != 3:
            filtered_ids.append(sample_id)

    
    poisoned_testset = PoisonedDatasetFolder(
        testset, 
        3,
        list(range(len(testset))), 
        poisoned_transform_index=0, 
        poisoned_target_transform_index=1, 
        reflection_cadidates=reflection_images,
        max_image_size=560, ghost_rate=0.49, alpha_b=0.1, offset=(0, 0), sigma=5, ghost_alpha=-1.)
    filtered_poisoned_testset = Subset(poisoned_testset,filtered_ids)
    return poisoned_trainset, filtered_poisoned_testset, trainset, testset
    
    
