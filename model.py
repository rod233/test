import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=13, kernel_size=9, stride=1, padding=4),# output shape(16, 28, 28)
            nn.ReLU(),
			nn.MaxPool2d(kernel_size=2) #output shape(16, 14, 14)
			)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=26, kernel_size=9, stride=1, padding=5),# output shape(32, 16, 16)
            nn.ReLU(),
			nn.MaxPool2d(kernel_size=2) #output shape(32, 8, 8)
			)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=52, kernel_size=9, stride=1, padding=4),# output shape(64, 8, 8)
            nn.ReLU(),
			nn.MaxPool2d(kernel_size=2) #output shape(64, 4, 4)
			)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=52, out_channels=104, kernel_size=9, stride=1, padding=4),# output shape(128, 4, 4)
            nn.ReLU(),
			nn.MaxPool2d(kernel_size=2) #output shape(128, 2, 2)
			)
        # 输出是10个数字
        self.fc = nn.Sequential(
            nn.Linear(104 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(64, 10)


    def forward(self,input):
        '''
        :param input: [batch_size,1,28*28] 我们获得的原始数据的样子，即input[0]=batchsize,input[1]=1,input[2]=28*28
        :return: 
        '''
        x = self.conv1(input)   
        x = self.dropout(x)		
        x = self.conv2(x) 
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.out(x)
        return out


        