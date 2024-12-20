file_list = 'webvid_list.txt'
anno_root_it = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons'
train_list_files = [f"{anno_root_it}/video/caption/videochat/train.json", f"{anno_root_it}/video/caption/webvid/train.json", f"{anno_root_it}/video/conversation/videochat1/train.json", f"{anno_root_it}/video/vqa/webvid_qa/train.json"]

import os
import json
from tqdm import tqdm

files = open(file_list).readlines()
files = [file.strip() for file in files]
f_missed = open('missing_files.txt', 'w')
for file in train_list_files:
    f = open(file, 'r')
    item_list = []
    data = json.load(f)
    for item in tqdm(data):
        video_id = item['video']
        if video_id not in files:
            f_missed.write(video_id + '\n')
        else:
            item_list.append(item)
    new_file = file.replace('train', 'train_new')
    with open(new_file, 'w') as f_new:
        json.dump(item_list, f_new)