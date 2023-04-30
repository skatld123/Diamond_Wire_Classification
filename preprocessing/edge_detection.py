import os
import cv2

def edgeDetection(directory):
    # directory 에 해당하는 파일들을 files로 전부 불러옴
    files = os.listdir(directory)

    for name in files:
        path = os.path.join(directory, name)
        img = cv2.imread(path)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        cv2.imwrite(os.path.join('whole_edge_detection_image', directory + '/' + name) , sobelx)
        print(os.path.join('whole_edge_detection_image', directory + '/' + name) )

path = 'whole_dataset'

edgeDetection(os.path.join(path, 'high'))
edgeDetection(os.path.join(path, 'medium'))
edgeDetection(os.path.join(path, 'low'))