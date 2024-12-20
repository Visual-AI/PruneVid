import os, json
anno_file = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/TRAIN_TEST/magic_jsons/video/vqa/ego_qa/train.json'
video_root_path = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/ego4d_data/split_videos'

annos = json.load(open(anno_file, 'r'))
for anno in annos:
    video_path = anno['video']
    video_path = os.path.join(video_root_path, video_path)
    if not os.path.exists(video_path):
        print(video_path)