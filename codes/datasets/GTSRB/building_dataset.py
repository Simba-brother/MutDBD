import os
import pandas as pd
import shutil

dataset_root_path = "/data/mml/backdoor_detect/dataset"
dataset_name = "GTSRB"
testset_dir = os.path.join(dataset_root_path, dataset_name, "testset")
os.makedirs(testset_dir, exist_ok=True)

df = pd.read_csv(os.path.join(dataset_root_path, dataset_name, "Test.csv"))
for row_id, row in df.iterrows():
    img_source_path = os.path.join(dataset_root_path, dataset_name, row["Path"]) 
    class_id = row["ClassId"]
    target_dir_path = os.path.join(testset_dir, str(class_id))
    os.makedirs(target_dir_path,exist_ok=True)
    image_name = row["Path"].split("/")[1]
    target_file_path = os.path.join(target_dir_path, image_name)
    shutil.copyfile(img_source_path, target_file_path)
print("build testset success")    
