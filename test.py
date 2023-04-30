#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model
author baiyu
"""
from dataset import MyDataset, getTrainingDataset
from model import cnn
import os
import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-dataset', type=str, default='resize_edge_detection_image', help='select dataset directory')
    parser.add_argument('-mode', type=str, default='test', help='select mode (train or test)')
    args = parser.parse_args()

    net = cnn.CNN()
    net = net.cuda()

    path = os.path.join(args.dataset, args.mode)

    data_list = getTrainingDataset(path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MyDataset(data_list, input_transform=transform)
    batch_size = 32
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloader)))

            image = image.cuda()
            label = label.cuda()
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(dataloader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(dataloader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))