# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author baiyu
"""
import cv2
import math
import numpy as np
from WireDataset import WireDataset, make_data_list, input_transform
from model.CNN import CNN
import os
import argparse
import time
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from model.ensemble_model import Ensemble

# 캐니 엣지 이미지 변환 및 전처리
canny_transform = transforms.Compose([
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), threshold1=50, threshold2=150)),  # 캐니 엣지 검출
    transforms.ToTensor(),  # 텐서로 변환

])

# 소벨 엣지 이미지 변환 및 전처리
sobel_transform = transforms.Compose([
    transforms.Lambda(lambda x: cv2.Sobel(np.array(x), cv2.CV_64F, 1, 0, ksize=3)),  # 소벨 엣지 검출
    transforms.ToTensor(),  # 텐서로 변환
])

def train(epoch):

    start = time.time()
    net.train()
    loss_list = []
    for batch_index, (images, labels) in enumerate(train_dataloader):

        batch_canny = None
        for image in images :
            image =  (image* 255).byte().numpy().transpose(1, 2, 0)
            canny = canny_transform(image).unsqueeze(0)
            sobel = sobel_transform(image).unsqueeze(0)
            
            # 배치에 텐서 추가
            if batch_canny is None:
                batch_canny = canny
                batch_sobel = sobel
            else:
                batch_canny = torch.cat([batch_canny, canny], dim = 0)
                batch_sobel = torch.cat([batch_sobel, sobel], dim = 0)
        
        labels = labels.cuda()
        images = images.float().cuda()
        canny = batch_canny.float().cuda()
        sobel = batch_sobel.float().cuda()
        optimizer.zero_grad()
        # print("types")
        # print("canny")
        # print(type(canny))
        # print("origin")
        # print(type(images))
        # print("sobel")
        # print(type(sobel))
        outputs = net(images, canny, sobel)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_dataloader.dataset)
        ))
        loss_list.append(loss.item())
        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    print()
    finish = time.time()
    
    avg = 0
    for loss in loss_list : 
        avg += loss
    print("avg = avg / len(train_dataloader)")
    print("{} = {} / {}".format(avg, avg, len(train_dataloader)))
    avg = avg / len(train_dataloader)
    with open('result_ensemble_train.txt', 'a') as f:
        f.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\n'.format(
            # 배치사이즈 개수에 맞는 loss 계산
            avg,
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_dataloader.dataset) 
        ))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    val_loss = 0.0 # cost function error
    val_correct = 0.0
    for (images, labels) in val_dataloader:
        batch_canny = None
        for image in images :
            image =  (image* 255).byte().numpy().transpose(1, 2, 0)
            canny = canny_transform(image).unsqueeze(0)
            sobel = sobel_transform(image).unsqueeze(0)
            
            # 배치에 텐서 추가
            if batch_canny is None:
                batch_canny = canny
                batch_sobel = sobel
            else:
                batch_canny = torch.cat([batch_canny, canny], dim = 0)
                batch_sobel = torch.cat([batch_sobel, sobel], dim = 0)
        canny = batch_canny.float().cuda()
        sobel = batch_sobel.float().cuda()
        images = images.float().cuda()
        labels = labels.cuda()

        outputs = net(images, canny, sobel)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        val_correct += preds.eq(labels).sum()

    finish = time.time()
    print('GPU INFO.....')
    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('val_length : {}'.format(len(val_dataloader.dataset)))
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        val_loss / len(val_dataloader.dataset),
        val_correct.float() / len(val_dataloader.dataset),
        finish - start
    ))
    with open('result_ensemble_val.txt', 'a') as f:
        f.write('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s\n'.format(
            epoch,
            # 배치사이즈 개수에 맞는 loss 계산
            val_loss / len(val_dataloader.dataset),
            val_correct.float() / len(val_dataloader),
            finish - start
        ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', val_loss / len(val_dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', val_correct.float() / len(val_dataloader.dataset), epoch)

    return val_correct.float() / len(val_dataloader.dataset)

if __name__ == '__main__':
    main_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-num_workers', type=int, default=4, help='torch DataLoader num_workers')
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')

    net = Ensemble()
    net = net.cuda()

    # 경로 선택 dataset/train
    train_path = os.path.join('dataset', 'original_train_test_val', 'train')
    # 경로 선택 dataset/val
    val_path = os.path.join('dataset', 'original_train_test_val', 'val')

    train_data_list = make_data_list(train_path)
    train_dataset = WireDataset(train_data_list, input_transform=input_transform())
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=args.num_workers)

    val_data_list = make_data_list(val_path)
    val_dataset = WireDataset(val_data_list, input_transform=input_transform())
    val_dataloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=args.num_workers)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 1, 426, 200)
    input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')
    best_acc = 0
    for epoch in range(1, settings.EPOCH + 1):
        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    # writer.close()

    main_finish = time.time()
    print('testing time consumed: {:.2f}s'.format(main_finish - main_start))
    
    