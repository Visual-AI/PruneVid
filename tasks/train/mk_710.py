annotation_file = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/classification/k710/train_new.json'
dst_path = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/k710'

import os, json, shutil
from tqdm import tqdm

f = open(annotation_file)
annotations = json.load(f)

for anno in tqdm(annotations):
    video_path = anno['video']
    video_name = os.path.basename(video_path)
    shutil.copyfile(video_path, dst_path + '/' + video_name)