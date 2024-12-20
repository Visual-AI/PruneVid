import shutil
import os

dataset_path = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/CLEVRER'
dir_list = os.listdir(dataset_path)

for dir in dir_list:
    dir_path = os.path.join(dataset_path, dir)
    file_list = os.listdir(dir_path)
    for file in file_list:
        file_path = os.path.join(dir_path, file)
        shutil.move(file_path, dataset_path)