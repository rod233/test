import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=9, stride=2, padding=0),# output shape(12, 10, 10)
			nn.MaxPool2d(kernel_size=5), #output shape(12, 2, 2)
			nn.Flatten(),
			nn.ReLU()
			)
        # 输出是10个数字
        self.out = nn.Linear(12 * 2 * 2, 10)


    def forward(self,input):
        '''
        :param input: [batch_size,1,28*28] 我们获得的原始数据的样子，即input[0]=batchsize,input[1]=1,input[2]=28*28
        :return: 
        '''
        x = self.conv1(input)       
		
        out = self.out(x)
        return out
