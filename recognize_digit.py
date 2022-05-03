import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MnistModel
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from numpy import *




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

train_dataloader = DataLoader(train_data, batch_size = 100, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size = 1000, shuffle=False)

device = torch.device("cuda")
model = MnistModel()#
model.to(device)

loss_func = torch.nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr=0.01)

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
        n = len(outputs)
        for i in range(n):
            img_list.append(outputs[i].argmax())
            labels_list.append(target[i])
        if batch_idx % 10 == 0:
            pre, acc, rec, f1 = indicator(img_list, labels_list)
            print("train_mode   precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre, acc, rec, f1))
            img_list = []
            labels_list = []

def test(model):
    model.eval()
    global img_list
    global labels_list
    
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
        
            outputs = model(data)
            outputs = outputs.cpu()
            target = target.cpu()
            n = len(outputs)
            for i in range(n):
                img_list.append(outputs[i].argmax())
                labels_list.append(target[i])		
        pre, acc, rec, f1 = indicator(img_list, labels_list)
        print("test_mode   precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre, acc, rec, f1))
        img_list = []
        labels_list = []
	
for epoch in range(epochs):
    train(model)
    test(model)

 #   with torch.no_grad():
 #       for imgs, labels in test_dataloader:
  #          outputs = model(imgs)
   #         loss = loss_func(outputs, labels)
    #        total_loss += loss
    #print("第{}次训练的loss：{}".format(i + 1, total_loss))

torch.save(model, "model.nn")