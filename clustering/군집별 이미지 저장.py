import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 이미지가 있는 폴더 경로
dir_path = '/root/dataset/edge_canny_train_test_val/'

# 각 클래스에 해당하는 이미지를 저장할 폴더 경로
cls1_path = '/root/dataset/cls1'
cls2_path = '/root/dataset/cls2'
cls3_path = '/root/dataset/cls3'

# 각 클래스에 해당하는 이미지들을 저장할 폴더 생성
os.makedirs(cls1_path, exist_ok=True)
os.makedirs(cls2_path, exist_ok=True)
os.makedirs(cls3_path, exist_ok=True)

# Canny 엣지 검출을 위한 임계값 설정
min_threshold = 50
max_threshold = 150

# 이미지 데이터셋 불러오기
img_list = []
for img in os.listdir(dir_path):
    image_path = os.path.join(dir_path, img)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 이미지 normalization
    # edges = cv2.Canny(image, min_threshold, max_threshold)
    # img_list.append(edges)
    img_list.append(image)

# 이미지 데이터셋을 1차원 벡터로 변환
X = np.array(img_list).reshape(len(img_list), -1)

# PCA를 이용하여 이미지 데이터 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# k-means-clustering 알고리즘을 적용하여 군집화
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_pca)

# 각 이미지가 어느 군집에 속하는지 확인
labels = kmeans.labels_

# 결과 출력
# for index, img in enumerate(os.listdir(dir_path)):
#     img_path = os.path.join(dir_path, img)
#     if labels[index] == 0:
#         cls_path = cls1_path
#     elif labels[index] == 1:
#         cls_path = cls2_path
#     else:
#         cls_path = cls3_path
#     # 이미지 저장
#     save_path = os.path.join(cls_path, img)
#     os.rename(img_path, save_path)

# 산점도 시각화
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.savefig('scatter_plot_canny.png')
