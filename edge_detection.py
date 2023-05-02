import os
import cv2
from os.path import join
from train_validation_test_split import make_dirs

def edgeDetection(directory):
    # directory 에 해당하는 파일들을 files로 전부 불러옴
    # files = os.listdir(directory)
    files = [f for f in os.listdir(directory) if os.path.isfile(join(directory, f))]
    for name in files:
        path = join(directory, name)    # dataset/whole/[name]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Sobel 필터 적용
        # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # sobelxy = sobelx + sobely
        # cv2.imwrite(join(directory, 'edge_detection_x', name) , sobelx)
        # cv2.imwrite(join(directory, 'edge_detection_y', name) , sobely)
        # cv2.imwrite(join(directory, 'edge_detection_xy', name) , sobelxy)

        # Scharr 필터 적용
        # scharrx = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        # scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        # scharrxy = scharrx + scharry
        # cv2.imwrite(join(directory, 'edge_detection_x', name) , scharrx)
        # cv2.imwrite(join(directory, 'edge_detection_y', name) , scharry)
        # cv2.imwrite(join(directory, 'edge_detection_xy', name) , scharrxy)
        
        # Canny 필터 적용
        cannyxy = cv2.Canny(img, 50, 200)
        cv2.imwrite(join(directory, 'edge_detection_Canny', name) , cannyxy)

path = 'dataset/whole'

# make_dirs(path, 'edge_detection_x')
# make_dirs(path, 'edge_detection_y')
make_dirs(path, 'edge_detection_Canny')

edgeDetection(path)