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

train_dataloader = DataLoader(train_data, batch_size = 32, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size = 32, shuffle=False)

model = MnistModel()#

loss_func = torch.nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr=0.001)

epochs = 10

BATCHSIZE = 10

img_list = []
labels_list = []


for i in range(epochs):
	for imgs, labels in train_dataloader:
		outputs = model(imgs)
		for i in range(32):
			img_list.append(outputs[i].argmax())
			labels_list.append(labels[i])
		loss = loss_func(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if len(img_list) == BATCHSIZE * 32:
			print("precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(precision_score(labels_list, img_list, average='macro'), accuracy_score(labels_list, img_list),recall_score(labels_list, img_list, average='macro'),f1_score(labels_list, img_list, average='macro')))
			img_list = []
			labels_list = []
	
	total_loss = 0
	with torch.no_grad():
		for imgs, labels in test_dataloader:
			outputs = model(imgs)
			loss = loss_func(outputs, labels)
			total_loss += loss
	print("第{}次训练的loss：{}".format(i + 1, total_loss))

torch.save(model, "model.nn")