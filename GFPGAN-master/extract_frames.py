import cv2
from tqdm import tqdm
import os
from os import path

outputPath = "/Users/greyleeee/Documents/GitHub/Wav2Lip-GFPGAN/outputs"
inputVideoPath = outputPath + '/result.mp4'
unProcessedFramesFolderPath = outputPath + '/frames'

# 创建目录（如果不存在的话）
if not os.path.exists(unProcessedFramesFolderPath):
    os.makedirs(unProcessedFramesFolderPath)

# 打开视频文件
vidcap = cv2.VideoCapture(inputVideoPath)
if not vidcap.isOpened():
    raise IOError(f"Error opening video file: {inputVideoPath}")

numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

# 逐帧读取视频并保存为图片
for frameNumber in tqdm(range(numberOfFrames)):
    success, image = vidcap.read()
    if not success:
        print(f"Warning: Failed to read frame {frameNumber}")
        continue
    cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4) + '.jpg'), image)

# 释放视频捕获对象
vidcap.release()
