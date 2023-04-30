import os
import cv2

# 이미지 디렉토리 선택
# img_dir = 'whole_dataset'
# img_dir = 'whole_edge_detection_image'
# img_dir = 'resize_dataset'
img_dir = 'resize_edge_detection_image'

whole = 'whole'

def my_train_test_split(dataset, label):
    data_path = os.path.join(dataset, label)
    files = os.listdir(data_path)
    images = []
    for name in files:
        path = os.path.join(data_path, name)
        img = cv2.imread(path)
        images.append(img)
    print(images[0].shape)

whole_dataset = os.path.join(img_dir, whole)

my_train_test_split(whole_dataset, 'high')
my_train_test_split(whole_dataset, 'medium')
my_train_test_split(whole_dataset, 'low')