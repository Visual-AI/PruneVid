data_root = '/root/paddlejob/workspace/env_run/data_afs_3/zhouhao14/intern/xiaohu/webvid/webvid'

import os
from tqdm import tqdm
f = open('webvid_list.txt', 'w')
dir_list = os.listdir(data_root)
for dir in tqdm(dir_list):
    dir_path = os.path.join(data_root, dir)
    file_list = os.listdir(dir_path)
    for file in file_list:
        file_path = os.path.join(dir_path, file)
        f.write(dir +'/'+ file + '\n')