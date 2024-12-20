annotation_file = "/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/classification/k710/train_new.json"
annotation_file_new = "/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/classification/k710/train_new_1.json"
file_list = open('k710_files_filter.txt', 'r').readlines()
file_list = [file.strip().split(' ') for file in file_list]
file_dict = {}
for file, path in file_list:
    file = file[:11].lower()
    file_dict[file] = path
import os
import json

annotations = json.load(open(annotation_file))
print('annoation length:', len(annotations))
annotations_new = []
count = 0
for anno in annotations:
    video_path = anno['video']
    video_path = video_path.split('/')[-1].split('.')[0]
    if len(video_path) > 15:
        video_path = video_path[:11]
    video_path = video_path.lower()
    if video_path in file_dict:
        # anno['video'] = file_dict[video_path.lower()]
        anno['video'] = anno['video'].split('/')[-1]
        annotations_new.append(anno)
    else:
        count += 1
json.dump(annotations_new, open(annotation_file_new, 'w'))
print('miss number:', count)
    # for file, file_path in file_list:
    #     if video_path in file:
    #         continue
    #     else:
    #         print(video_path)