import torch
import torch.nn as nn
import torch.nn.functional as F
import model.cnn_network as cn

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.pipeline_1 = cn.Network()
        self.pipeline_2 = cn.Network()
        self.pipeline_3 = cn.Network()
        self.conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=64*12*26, out_features=1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=3)
        

    def forward(self, origin, canny, sobel):
        canny = self.pipeline_1(canny)
        sobel = self.pipeline_2(sobel)
        orign = self.pipeline_3(origin)
        # print("canny")
        # print(canny.shape)
        # print("sobel")
        # print(sobel.shape)
        # print("origin")
        x = torch.cat((canny, sobel, orign), dim = 1)
        x = self.conv(x)
        x = self.pool(x)
        x= x.view(x.size(0), -1)
        x = self.fc1(x)
        x= self.dropout(x)
        x = self.fc2(x)
        x= self.dropout(x)
        x = self.fc3(x)
        out= F.softmax(x, dim=1)
        return out

def getModel():
    return Ensemble()