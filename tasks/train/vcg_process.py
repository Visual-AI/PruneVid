anno_file = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/conversation/videochatgpt/train_new.json'
anno_new_file = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/conversation/videochatgpt/train_new_1.json'

data_root = '//root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/panda'
import os, json
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm


miss_count = 0
count = 0
# annos = json.load(open(anno_file))
# annos_new = []
# for anno in tqdm(annos):
#     video_path = os.path.join(data_root, anno['video'])
#     if not os.path.exists(video_path):
#         continue
#     try:
#         vr = VideoReader(video_path, ctx=cpu(0))
#         annos_new.append(anno)
#     except:
#         count += 1
# json.dump(annos_new, open(anno_new_file, 'w'))
# print(count)

files = os.listdir(data_root)
for file in tqdm(files):
    video_path = os.path.join(data_root, file)
    try:
        count += 1
        vr = VideoReader(video_path, ctx=cpu(0))
    except:
        miss_count += 1
        print(video_path)
print(miss_count, count)