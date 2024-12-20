import subprocess
import os

# 源文件夹路径
source_folder = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/VideoQA/TGIF_QA/video_gif'
target_folder = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/VideoQA/TGIF_QA/videos_mp4'
# 包含文件名的文本文件路径
file_list_path = 'not_have.txt'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 读取文件列表并转换
with open(file_list_path, 'r') as file_list:
    for line in file_list:
        # 获取去除前后空白字符的文件名
        gif_filename = line.strip()
        # 源文件完整路径
        source_path = os.path.join(source_folder, gif_filename)
        # 目标文件完整路径，假设输入文件名格式正确，并将后缀替换为.mp4
        target_path = os.path.join(target_folder, os.path.splitext(gif_filename)[0] + '.mp4')

        # 构建ffmpeg命令
        cmd = ['ffmpeg', '-i', source_path, '-movflags', 'faststart', target_path]
        
        # 执行命令
        try:
            subprocess.run(cmd, check=True)
            print(f'Successfully converted {gif_filename} to MP4.')
        except subprocess.CalledProcessError as e:
            print(f'Failed to convert {gif_filename}. Error: {e}')

print('All files have been processed.')