from tensorboardX import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MnistModel
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import ASGD
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import Adamax
from torch.optim import RMSprop
from torch.optim import Rprop
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from numpy import *
from torchvision.utils import make_grid


logger = SummaryWriter(log_dir="data/log")

train_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

# 将测试数据压缩到0-1
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets

train_dataloader = DataLoader(train_data, batch_size = 16, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle=False)

device = torch.device("cuda")
model = MnistModel()#
model.to(device)

loss_func = torch.nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr=0.001, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

epochs = 10

img_list = []
labels_list = []

TP = [0] * 10
FP = [0] * 10
FN = [0] * 10

 

def indicator(pred_list, true_list):
    pre = precision_score(true_list, pred_list, average='micro') 
    acc = accuracy_score(true_list, pred_list)
    rec = recall_score(true_list, pred_list, average='micro')
    f1 = f1_score(true_list, pred_list, average='micro')
    return pre, acc, rec, f1	
	
def init_weights(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

model.apply(init_weights)

def train(model):
    model.train()
    global img_list
    global labels_list
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_func(outputs, target)
        loss.backward()
        optimizer.step()
        outputs = outputs.cpu()
        target = target.cpu()
   #     print(outputs)
        n = len(outputs)
        for i in range(n):
            img_list.append(outputs[i].argmax())
            labels_list.append(target[i])
        if batch_idx % 100 == 0:
     #       test_predict = MyConvNet(test_data_x)
     #       predict_idx = torch.max(test_predict, 1)
            pre, acc, rec, f1 = indicator(img_list, labels_list)
     #       print("train_mode   precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre, acc, rec, f1))
            logger.add_scalar("train loss", loss.item() ,global_step=batch_idx)
            logger.add_scalar("test accuracy", acc.item() ,global_step=batch_idx)
            logger.add_scalar("test precision", pre.item() ,global_step=batch_idx)
            logger.add_scalar("test recall", rec.item() ,global_step=batch_idx)
            logger.add_scalar("test F1", f1.item() ,global_step=batch_idx)
            img = make_grid(data, nrow=12)
            logger.add_image("train image sample", img, global_step=batch_idx)
            for name, param in model.named_parameters():
                param = param.cpu()
                logger.add_histogram(name, param.data.numpy(), global_step=batch_idx)
            img_list = []
            labels_list = []

def test(model, epoch):
    model.eval()
    global img_list
    global labels_list
    max_pre = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
        
            outputs = model(data)
            loss = loss_func(outputs, target)
            total_loss += loss
            outputs = outputs.cpu()
            target = target.cpu()
            n = len(outputs)
            for i in range(n):
                img_list.append(outputs[i].argmax())
                labels_list.append(target[i])		
        pre, acc, rec, f1 = indicator(img_list, labels_list)
        print("test_mode   precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre, acc, rec, f1))
        print("第{}次训练的Loss:{}".format(epoch + 1, total_loss))
        img_list = []
        labels_list = []
 #       if pre > max_pre:
  #          max_pre = pre
   #         torch.save(model, "model.nn")
	
for epoch in range(epochs):
    train(model)
    test(model, epoch)
    scheduler.step()

 #   with torch.no_grad():
 #       for imgs, labels in test_dataloader:
  #          outputs = model(imgs)
   #         loss = loss_func(outputs, labels)
    #        total_loss += loss
    #print("第{}次训练的loss：{}".format(i + 1, total_loss))

torch.save(model, "model.nn")