import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 이미지 데이터셋 디렉토리 경로
dataset_path = "/root/dataset/edge_SobelXY_train_test_val"

# 이미지 파일 목록 가져오기
image_files = os.listdir(dataset_path)

# 이미지 데이터를 담을 리스트 초기화
image_data_2d = []
image_data = []
file_names = []  # 파일 이름을 저장할 리스트 초기화

# 이미지 파일 불러와서 데이터 추출
for file in image_files:
    image_path = os.path.join(dataset_path, file)
    image = Image.open(image_path)  # 이미지 파일 열기
    image = image.resize((100, 100))  # 이미지 크기 조정
    image = np.array(image)  # 이미지를 numpy 배열로 변환
    image_data.append(image)
    image_data_2d.append(image.flatten())  # 이미지 데이터를 1차원으로 펼쳐서 리스트에 추가
    file_names.append(file)  # 파일 이름 저장

# 이미지 데이터를 numpy 배열로 변환
image_data = np.array(image_data)
image_data_2d = np.array(image_data_2d)

# 이미지 데이터를 3차원으로 변환
image_reshaped = image_data.reshape(len(image_data), -1)

# PCA를 사용하여 이미지 데이터를 3차원으로 축소
pca = PCA(n_components=3)
image_3d = pca.fit_transform(image_reshaped)

# K-means 클러스터링 적용
kmeans = KMeans(n_clusters=3)
kmeans.fit(image_data_2d)

# 클러스터링 결과 가져오기
labels = kmeans.labels_

# 클러스터링 결과 시각화 및 이미지 저장
for group in range(3):
    cluster_images = image_data[labels == group]
    cluster_file_names = np.array(file_names)[labels == group]  # 해당 클러스터의 파일 이름 추출
    
    n = len(cluster_images)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    ratio = 1
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols * ratio, rows * ratio), squeeze=False)

    cnt_low = 0
    cnt_high = 0
    cnt_med = 0
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(cluster_images[i * 10 + j], cmap='gray_r')
                if "low" in cluster_file_names[i * 10 + j] : 
                    cnt_low += 1
                elif "high" in cluster_file_names[i * 10 + j] : 
                    cnt_high += 1
                elif "med" in cluster_file_names[i * 10 + j] : 
                    cnt_med += 1
                axs[i, j].set_title(cluster_file_names[i * 10 + j][0:3])  # 파일 이름을 타이틀로 설정
                axs[i, j].title.set_size(6)
            axs[i, j].axis('off')
    fig.suptitle(f"Cluster {group+1} : Low {cnt_low}, Med {cnt_med}, High {cnt_high}")
    plt.savefig(f"cluster_images_{group+1}.png")  # 각 그룹 이미지 저장
    plt.close()

# 클러스터링 결과 산점도 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 산점도 그리기
ax.scatter(image_3d[:, 0], image_3d[:, 1], image_3d[:, 2], c=labels)
print(labels)
# 데이터 포인트의 파일 이름 표시
for i, label in enumerate(labels):
    ax.text(image_3d[i, 0], image_3d[i, 1], image_3d[i, 2], file_names[i][0:1], fontsize=8)
    # ax.text(image_3d[i, 0], image_3d[i, 1], image_3d[i, 2], file_names[i][-11:], fontsize=8)

# 그래프 표시
plt.savefig("clustering_scatter.png")  # 클러스터링 결과 산점도 저장
plt.show()