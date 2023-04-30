import os
import cv2
from sklearn.model_selection import train_test_split

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
    labels = []
    for name in files:
        path = os.path.join(data_path, name)
        img = cv2.imread(path)
        images.append(img)
        labels.append(label)
    train_img, test_img, train_label, test_label = train_test_split(images, labels, random_state=42)
    for idx, img in enumerate(train_img):
        cv2.imwrite(os.path.join(img_dir + '/train/' + label, 'train_' + str(idx) + '.bmp') , img)
    for idx, img in enumerate(test_img):
        cv2.imwrite(os.path.join(img_dir + '/test/' + label, 'test_' + str(idx) + '.bmp') , img)

whole_dataset = os.path.join(img_dir, whole)

my_train_test_split(whole_dataset, 'high')
my_train_test_split(whole_dataset, 'medium')
my_train_test_split(whole_dataset, 'low')