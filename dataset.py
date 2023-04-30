import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_list, input_transform):
        self.data_list = data_list
        self.input_transform = input_transform
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 데이터셋에서 idx 번째 데이터 로드
        image, label = self.data_list[idx]
        label = int(label)
        if self.input_transform:
            image = self.input_transform(image)

        return image, label

def getTrainingDataset(directory, label='high', label2='medium', label3='low'):
    data_path = os.path.join(directory, label)
    files = os.listdir(data_path)
    data_list = []
    for name in files:
        path = os.path.join(data_path, name)
        img = cv2.imread(path)
        data_list.append((img, 0))

    data_path = os.path.join(directory, label2)
    files = os.listdir(data_path)
    for name in files:
        path = os.path.join(data_path, name)
        img = cv2.imread(path)
        data_list.append((img, 1))
    
    data_path = os.path.join(directory, label3)
    files = os.listdir(data_path)
    for name in files:
        path = os.path.join(data_path, name)
        img = cv2.imread(path)
        data_list.append((img, 2))
    return data_list
