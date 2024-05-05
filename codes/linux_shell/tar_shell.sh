
#!/bin/bash
 
# 要解压的tar文件所在目录
tar_dir="/data/mml/dataset/ILSVRC2012_img_train"
 
# 遍历tar文件所在目录
for tar_file in "$tar_dir"/*.tar; do
    # 获取文件夹名称，通常为tar文件名去掉后缀
    folder_name="${tar_file%.tar}"
    # 创建对应的文件夹
    mkdir -p "$folder_name"
    # 解压tar文件到对应的文件夹内
    tar -xf "$tar_file" -C "$folder_name"

for tar_file in "$tar_dir"/*.tar; do
    rm "$tar_file"
done

