import cv2
import os
from tqdm import tqdm

# 定义路径
outputPath = "/Users/greyleeee/Documents/GitHub/Wav2Lip-GFPGAN/outputs"
restoredFramesPath = outputPath + '/restored_imgs/'
processedVideoOutputPath = outputPath

# 确保输出目录存在
if not os.path.exists(processedVideoOutputPath):
    os.makedirs(processedVideoOutputPath)

# 获取已处理帧的列表
dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

# 处理批次
batch = 0
batchSize = 300

# 合成视频
for i in tqdm(range(0, len(dir_list), batchSize)):
    img_array = []
    start, end = i, i + batchSize
    print("Processing frames from", start, "to", end)
    
    # 读取图像并添加到 img_array
    for filename in tqdm(dir_list[start:end]):
        filename = restoredFramesPath + filename
        img = cv2.imread(filename)
        if img is None:
            continue
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    # 写入视频文件
    out = cv2.VideoWriter(processedVideoOutputPath + '/batch_' + str(batch).zfill(4) + '.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    batch += 1
    
    for img in img_array:
        out.write(img)
    
    out.release()
