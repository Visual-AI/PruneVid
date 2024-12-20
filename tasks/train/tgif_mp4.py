import os
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from decord import VideoReader
from decord import cpu, gpu


# 源文件夹和目标文件夹
source_folder = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/VideoQA/TGIF_QA/video_gif'
target_folder = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/VideoQA/TGIF_QA/videos_mp4'
used_list = 'tgif_used.txt'
corrupt_list = 'tgif_corrupt.txt'
f_corrupt = open(corrupt_list, 'w')
# 读取使用的文件列表
with open(used_list, 'r') as f:
    used_files = [line.strip() for line in f.readlines()]

count = 0
not_count = 0
miss_list = []
for file in tqdm(used_files):
    # file_new = file.replace('.gif', '.mp4')
    video_path = os.path.join(target_folder, file)
    if os.path.exists(video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except:
            count += 1
            miss_list.append(file)
            print('Error processing {}'.format(file))
            f_corrupt.write(file + '\n')
    else:
        not_count += 1
print(count, not_count)
    # print(os.path.join(source_folder, file))
# print(count, len(used_files))
# miss_list = []

# def convert_gif_to_mp4(file):
#     if file.endswith('.gif'):
#         source_path = os.path.join(source_folder, file)
#         target_path = os.path.join(target_folder, file.replace('.gif', '.mp4'))

#         if os.path.exists(target_path):
#             return f'{file} already converted. Skipping...'
#         try:
#             clip = VideoFileClip(source_path)
#             clip.write_videofile(target_path, codec="libx264", fps=24)
#             clip.close()
            
#             return f'Saved {file.replace(".gif", ".mp4")} to {target_folder}'
#         except Exception as e:
#             print('Error processing {}'.format(file), e, sep='\n')
#             # raise e
#             return f'Error processing {file}: {e}'
#     else:
#         return f'{file} is not a GIF. Skipping...'

# # 设置线程池的最大线程数
# max_threads = 8

# with ThreadPoolExecutor(max_workers=max_threads) as executor:
#     # 使用executor.map来并行处理任务
#     # 注意：如果你想在任务执行时保持进度条更新，可能需要使用executor.submit和as_completed
#     futures = [executor.submit(convert_gif_to_mp4, file) for file in used_files]
    
#     # 为了展示进度条，我们使用as_completed来获取已完成的future
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         print(future.result())

# print("所有视频处理完成！")