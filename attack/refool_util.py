
import os
import cv2
def get_reflection_images():
    # 存储反射照片
    reflection_images = []
    # URL：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    # "/data/ganguanhao/datasets/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set
    reflection_data_dir = "/data/mml/backdoor_detect/dataset/VOCdevkit/VOC2012/JPEGImages" 

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
        
    # reflection image dir下所有的img path
    reflection_image_path = os.listdir(reflection_data_dir)
    # 读出来前200个reflection img
    reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
    return reflection_images
