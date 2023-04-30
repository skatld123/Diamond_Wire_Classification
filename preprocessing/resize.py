import os
import cv2

# 이미지 디렉토리 선택
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
        img = cv2.resize(img, (426, 200))
        images.append(img)
    for idx, img in enumerate(images):
        cv2.imwrite(os.path.join(data_path, label + str(idx) + '.bmp') , img)
        print(img.shape)

whole_dataset = os.path.join(img_dir, whole)

my_train_test_split(whole_dataset, 'high')
my_train_test_split(whole_dataset, 'medium')
my_train_test_split(whole_dataset, 'low')