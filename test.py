#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model
author baiyu
"""
from WireDataset import WireDataset, make_data_list, input_transform
from model.CNN import CNN
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from conf import settings

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=32, help='batch size')
    parser.add_argument('-num_workers', type=int, default=8, help='torch DataLoader num_workers')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    args = parser.parse_args()

    net = CNN()
    net = net.cuda()

    path = os.path.join('dataset', 'test')

    data_list = make_data_list(path)
    dataset = WireDataset(data_list, input_transform=input_transform())
    dataloader = DataLoader(dataset, batch_size=args.b, shuffle=True, num_workers=args.num_workers)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_2 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(dataloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(dataloader)))
            image = image.cuda()
            label = label.cuda()
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
            output = net(image)
            _, pred = output.topk(k=min(2, output.size(1)), dim=1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            #compute top1
            correct_1 += correct[:, :1].sum()
            #compute top2
            correct_2 += correct[:, :2].sum()
    finish = time.time()

    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(dataloader.dataset))
    print("Top 2 err: ", 1 - correct_2 / len(dataloader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    print('testing time consumed: {:.2f}s'.format(finish - start))
