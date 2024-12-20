import cv2

# 视频文件路径
video_path = '/root/paddlejob/workspace/env_run/output/xiaohu/data/video_vlm/PLLaVA/DATAS/Video_ChatGPT/v_Z0eBz6QsI-c.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # 显示当前帧
    # cv2.imshow('frame', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放Capture对象
cap.release()
cv2.destroyAllWindows()