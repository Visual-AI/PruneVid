dataset_path = {
   #  'k400': '/root/paddlejob/workspace/env_run/output/xiaohu/data/k400/train',
   #  'k600': '/root/paddlejob/workspace/env_run/output/xiaohu/data/k600/Kinetics600/videos',
   #  'k700': '/root/paddlejob/workspace/env_run/data_afs_3/zhouhao14/intern/xiaohu/k700_dir/Kinetics_700/videos/'
   'k710': '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/k710'
}

import os
from tqdm import tqdm

f = open('k710_files_filter.txt', 'w')
for dataset, path in dataset_path.items():
   # dir_list = os.listdir(path)
   # for dir in tqdm(dir_list):
   #    dir_path = os.path.join(path, dir)
   file_list = os.listdir(path)
   for file in file_list:
      file_path = os.path.join(path, file)
      f.write(file+' '+file_path+'\n')