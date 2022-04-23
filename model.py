import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(in_features=28*28,out_features=28)
        # 输出是10个数字
        self.fc2 = nn.Linear(28,10)


    def forward(self,input):
        '''
        :param input: [batch_size,1,28*28] 我们获得的原始数据的样子，即input[0]=batchsize,input[1]=1,input[2]=28*28
        :return: 
        '''
        x = input.view(input.size(0),28*28) 
        #进行全连接操作
        x = self.fc1(x)
        #使用激活函数处理数据，不会使形状发生变化
        x = F.relu(x)
        #输出层
        out = self.fc2(x)
        return out
