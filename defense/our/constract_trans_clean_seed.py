
import os
import random
import shutil
from pathlib import Path


cifar_className_to_imageNet_class_name = {
    "airplane": [
        "n02690373",  # airliner
        "n04552348",  # warplane
    ],
    "automobile": [
        "n03100240",  # convertible
        "n03594945",  # jeep
        "n03670208",  # limousine
        "n03770679",  # minivan
        "n03777568",  # Model T
        "n04037443",  # racer
    ],
    "bird": [
        "n01530575",  # brambling
        "n01531178",  # goldfinch
        "n01532829",  # house finch
        "n01537544",  # indigo bunting
        "n01558993",  # robin
        "n01560419",  # bulbul
        "n01580077",  # jay
        "n01582220",  # magpie
        "n01592084",  # chickadee
    ],
    "cat": [
        "n02123045",  # tabby
        "n02123159",  # tiger cat
        "n02123394",  # Persian cat
        "n02123597",  # Siamese cat
        "n02124075",  # Egyptian cat
    ],
    "deer": [
        "n02417914",  # ibex
        "n02422106",  # hartebeest
        "n02422699",  # impala
        "n02423022",  # gazelle
        "n02415577",  # bighorn (deer-like ungulate)
    ],
    "dog": [
        "n02085620",  # Chihuahua
        "n02099601",  # golden retriever
        "n02106662",  # German shepherd
        "n02107142",  # Doberman
        "n02110341",  # dalmatian
        "n02109047",  # Great Dane
        "n02108551",  # Tibetan mastiff
        "n02109525",  # Saint Bernard
    ],
    "frog": [
        "n01641577",  # bullfrog
        "n01644373",  # tree frog
        "n01644900",  # tailed frog
    ],
    "horse": [
        "n02389026",  # sorrel (a horse)
        "n02391049",  # zebra (horse-family; included to increase variety)
    ],
    "ship": [
        "n03095699",  # container ship
        "n03673027",  # liner
        "n02687172",  # aircraft carrier
        "n03947888",  # pirate
        "n04552696",  # warship
    ],
    "truck": [
        "n03417042",  # garbage truck
        "n03930630",  # pickup
        "n04467665",  # trailer truck
        "n04461696",  # tow truck
    ],
}


def main():
    '''
    帮我实现下面的功能:

    基于"cifar_className_to_imageNet_class_name"字典，帮我构建一个新数据集，新数据集保存目录在"target_dir"
    该数据集要求以字典的key为分类目录，每个key分类下共包含10张图像，这10张图像就随机来自对应的imagenet的目录下的图像。imagenet数据集目录
    在"imagenet_train_dir"。最后数据集的后缀都改为png
    '''

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Process each CIFAR class
    for cifar_class, imagenet_classes in cifar_className_to_imageNet_class_name.items():
        print(f"Processing {cifar_class}...")

        # Create class directory
        class_dir = os.path.join(target_dir, cifar_class)
        os.makedirs(class_dir, exist_ok=True)

        # Collect all available images from corresponding ImageNet classes
        all_images = []
        for imagenet_class in imagenet_classes:
            imagenet_class_dir = os.path.join(imagenet_train_dir, imagenet_class)
            if os.path.exists(imagenet_class_dir):
                images = [os.path.join(imagenet_class_dir, f)
                         for f in os.listdir(imagenet_class_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
                all_images.extend(images)

        # Randomly sample 10 images
        if len(all_images) >= 10:
            selected_images = random.sample(all_images, 10)
        else:
            print(f"Warning: Only {len(all_images)} images found for {cifar_class}, using all available")
            selected_images = all_images

        # Copy images and change extension to .png
        for idx, img_path in enumerate(selected_images):
            try:
                output_path = os.path.join(class_dir, f"{idx:03d}.png")
                shutil.copy2(img_path, output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"  Saved {len(selected_images)} images to {class_dir}")



if __name__ == "__main__":

    imagenet_train_dir = "/data/mml/dataset/ImageNet_2012/train"
    target_dir = "/data/mml/backdoor_detect/dataset/CIFAR10_transfer_cleanSeed"
    main()