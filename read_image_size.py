import os
import cv2

# 디렉토리 선택
img_dir = 'dataset/train'
files = os.listdir(img_dir)
for name in files:
    path = os.path.join(img_dir, name)
    img = cv2.imread(path)
    print(img.shape)