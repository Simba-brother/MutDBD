
import copy
from torchvision import transforms
from torchvision.datasets import DatasetFolder

class Add_IAD_DatasetFolderTrigger():
    """Add IAD trigger to DatasetFolder images.
    """

    def __init__(self, modelG, modelM):
         self.modelG = modelG
         self.modelM = modelM

    def __call__(self, img):
        # 允许一个类的实例像函数一样被调用
        """Get the poisoned image..
        img: shap:CHW,type:Tensor
        """
        
        # 添加一个维度索引构成BCHW
        imgs = img.unsqueeze(0) # 增加一个B维度
        # G model生成pattern
        patterns = self.modelG(imgs)
        # 对pattern normalize一下
        patterns = self.modelG.normalize_pattern(patterns)
        # 获得masks
        masks_output = self.modelM.threshold(self.modelM(imgs))
        # inputs, patterns, masks => bd_inputs
        bd_imgs = imgs + (patterns - imgs) * masks_output 
        # 压缩一个维度
        bd_img = bd_imgs.squeeze(0) # 不会replace
        bd_img= bd_img.detach()
        return bd_img

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_ids:list,
                 modelG,
                 modelM
                 ):
        super(PoisonedDatasetFolder, self).__init__(
            # 数据集文件夹位置
            benign_dataset.root, # 数据集文件夹 /data/mml/backdoor_detect/dataset/cifar10/train
            # 数据集直接加载器
            benign_dataset.loader, # cv2.imread
            # 数据集扩展名
            benign_dataset.extensions, # .png
            # 数据集transform
            benign_dataset.transform, # 被注入到self.transform
            # 数据集标签transform
            benign_dataset.target_transform, # 对label进行transform
            None)
        self.poisoned_set = poisoned_ids
        # Add trigger to images
        # 注意在调用父类（DatasetFolder）构造时self.transform = benign_dataset.transform
        if self.transform is None:
            self.poisoned_transform = transforms.Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform) # Compose()的深度拷贝    
        # 中毒转化器为在普通样本转化器前再加一个AddDatasetFolderTrigger
        self.poisoned_transform.transforms.append(Add_IAD_DatasetFolderTrigger(modelG, modelM))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = transforms.Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.append(ModifyTarget(y_target))

    def __getitem__(self, index):
        # DatasetFolder 必须要有迭代
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 父类（DatasetFolder）属性
        sample = self.loader(path) # self.loader也是调用父类构造时注入的
        isPoisoned = False
        if index in self.poisoned_set: # self.poisoned_set 本类构造注入的
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
            isPoisoned = True
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target, isPoisoned