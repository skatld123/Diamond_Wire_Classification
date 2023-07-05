# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author baiyu
"""
from WireDataset import WireDataset, make_data_list, input_transform
from model.CNN_feature_visualize import CNN
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

def train():
    net.train()
    for batch_index, (images, _) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = net(images)
        feature_map = outputs.detach().numpy()[0]
        fig, axarr = plt.subplots(4, 4, figsize=(6, 6))
        for idx in range(16):
            ax = axarr[int(idx / 4), idx % 4]
            ax.imshow(feature_map[idx], cmap='gray')
            ax.axis('off')
        # 경로 설정
        plt.savefig(f'feature/origin/conv/feature_map_{batch_index}.png')

if __name__ == '__main__':
    main_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=32, help='batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-num_workers', type=int, default=8, help='torch DataLoader num_workers')
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')

    net = CNN()

    dataset_path = [
        'original_train_test_val',
        # 'edge_canny_train_test_val',
        # 'edge_Laplacian_train_test_val',
        # 'edge_ScharrXY_train_test_val',
        # 'edge_SobelXY_train_test_val'
    ]
    train_datasets = []
    val_datasets = []

    for path in dataset_path:
        train_path = os.path.join('dataset', path, 'train')
        train_data_list = make_data_list(train_path)
        train_dataset = WireDataset(train_data_list, input_transform=input_transform())
        train_datasets.append(train_dataset)

    concat_train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(concat_train_dataset, shuffle=True, num_workers=args.num_workers)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train()

    main_finish = time.time()
    print('testing time consumed: {:.2f}s'.format(main_finish - main_start))