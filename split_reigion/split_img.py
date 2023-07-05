import cv2
import os

# 이미지 불러오기
dir_path = '/root/dataset/medium/'
img_list = os.listdir(dir_path)
for index, img in enumerate(img_list) :
    image_path = os.path.join(dir_path, img)
    print(image_path)
    image = cv2.imread(image_path)

    # 이미지 4분할
    height, width = image.shape[:2]
    print(str(height) + " : " + str(width))
    sub_width = width // 4

    for i in range(4):
        sub_image = image[:height, i*sub_width:(i+1)*sub_width]
        
        # 저장할 디렉토리 생성
        save_dir = '/root/dataset/split_medium'
        os.makedirs(save_dir, exist_ok=True)
        
        # 저장할 파일명 생성
        save_path = os.path.join(save_dir, 'medium_%03d_%03d.jpg' % (index, i))
        
        # 이미지 저장
        cv2.imwrite(save_path, sub_image)
