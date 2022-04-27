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

train_dataloader = DataLoader(train_data, batch_size = 10, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size = 10, shuffle=False)

model = MnistModel()#

loss_func = torch.nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr=0.001)

epochs = 10

BATCHSIZE = 10

img_list = []
labels_list = []

TP = [0] * 10
FP = [0] * 10
FN = [0] * 10
for i in range(epochs):
	for imgs, labels in train_dataloader:
		outputs = model(imgs)
		for i in range(10):
			img_list.append(outputs[i].argmax())
			labels_list.append(labels[i])
		loss = loss_func(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if len(img_list) == BATCHSIZE * 10:
			print("pred:{}, true:{}".format(img_list, labels_list))
			for j in range(len(img_list)):
				if img_list[j] == 0:
					if img_list[j] == labels_list[j]:
						TP[0] += 1
					else:
						FP[0] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 1:
					if img_list[j] == labels_list[j]:
						TP[1] += 1
					else:
						FP[1] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 2:
					if img_list[j] == labels_list[j]:
						TP[2] += 1
					else:
						FP[2] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 3:
					if img_list[j] == labels_list[j]:
						TP[3] += 1
					else:
						FP[3] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 4:
					if img_list[j] == labels_list[j]:
						TP[4] += 1
					else:
						FP[4] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 5:
					if img_list[j] == labels_list[j]:
						TP[5] += 1
					else:
						FP[5] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 6:
					if img_list[j] == labels_list[j]:
						TP[6] += 1
					else:
						FP[6] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 7:
					if img_list[j] == labels_list[j]:
						TP[7] += 1
					else:
						FP[7] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 8:
					if img_list[j] == labels_list[j]:
						TP[8] += 1
					else:
						FP[8] += 1
						FN[labels_list[j]] += 1
				elif img_list[j] == 9:
					if img_list[j] == labels_list[j]:
						TP[9] += 1
					else:
						FP[9] += 1
						FN[labels_list[j]] += 1
			print("TP:{}, FP:{}, FN:{}".format(TP, FP, FN))
			accu = [0] * 10
			prec = [0] * 10
			recall = [0] * 10
			for x in range(10):
				if TP[x] > 0:
					accu[x] = TP[x] / (TP[x] + FP[x] + FN[x])
					prec[x] = TP[x] / (TP[x] + FP[x])
					recall[x] = TP[x] / (TP[x] + FN[x])
			acc_totall = mean(accu)
			pre_totall = mean(prec)
			rec_totall = mean(recall)
	#		acc = 0
	#		pre = 0
	#		rec = 0	
	#		acc_num = 0
	#		pre_num = 0
	#		rec_num = 0
	#		for a in range(len(accu)):
	#			if accu[a] > 0:
	#				acc += accu[a]
	#				acc_num += 1
	#		for b in range(len(prec)):
	#			if prec[b] > 0:
	#				pre += prec[b]
	#				pre_num += 1
	#		for c in range(len(recall)):
	#			if recall[c] > 0:
	#				rec += recall[c]
	#				rec_num += 1
	#		acc_totall = acc / acc_num
	#		pre_totall = pre / pre_num
	#		rec_totall = rec / rec_num
	#		for x in range(10):
	#			TP_totall += TP[x]
	#			FP_totall += FP[x]
	#			FN_totall += FN[x]
	#		print("TP:{}, FP:{}, FN:{}".format(TP_totall, FP_totall, FN_totall))
	#		accu = TP_totall / (TP_totall + FP_totall)
	#		prec = TP_totall / (TP_totall + FP_totall)
	#		recall = TP_totall / (TP_totall + FN_totall)
			F1 = 2 * (pre_totall * rec_totall) / (pre_totall + rec_totall)
			print("precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre_totall, acc_totall, rec_totall, F1))
			TP = [0] * 10
			FP = [0] * 10
			FN = [0] * 10
			
			print("precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(precision_score(labels_list, img_list, average='micro'), accuracy_score(labels_list, img_list),recall_score(labels_list, img_list, average='macro'),f1_score(labels_list, img_list, average='macro')))
			img_list = []
			labels_list = []
			print("*****************************************")
	
	total_loss = 0
	with torch.no_grad():
		for imgs, labels in test_dataloader:
			outputs = model(imgs)
			loss = loss_func(outputs, labels)
			total_loss += loss
	print("第{}次训练的loss：{}".format(i + 1, total_loss))

torch.save(model, "model.nn")