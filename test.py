import PIL
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

device = torch.device("cuda")
model = torch.load("model.nn")
model.to(device)

class MyTestSet(Dataset):
	def __init__(self):
		self.transform = transforms.Compose(
			[
				transforms.ToTensor()
				#transforms.Normalize()
			]
		)
		self.data = []
		self.labels = []
		for digit in range(10):
			for i in range(50):
				path = "./verify_data/test_data/{}/{}.bmp".format(digit, i)
				img = Image.open(path)
				img = self.transform(img)
				self.data.append(torch.unsqueeze(img[3], dim=0))
				self.labels.append(digit)
	
	def __len__(self):
		return len(self.data)
		
	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]
		

test_set = MyTestSet()
test_dataloader = DataLoader(test_set, batch_size=5, shuffle=False)

y = 0
n = 0
img_list = []
labels_list = []

def indicator(pred_list, true_list):
    pre = precision_score(true_list, pred_list, average='micro') 
    acc = accuracy_score(true_list, pred_list)
    rec = recall_score(true_list, pred_list, average='micro')
    f1 = f1_score(true_list, pred_list, average='micro')
    return pre, acc, rec, f1	

for data, label in test_dataloader:
    data, label = data.to(device), label.to(device)
    outputs = model(data)
    outputs = outputs.cpu()
    label = label.cpu()
    n = len(outputs)
    for i in range(n):
        img_list.append(outputs[i].argmax())
        labels_list.append(label[i])
pre, acc, rec, f1 = indicator(img_list, labels_list)
print("verify_mode   precision_score:{}, accuracy_score:{},recall_score:{},F1 score:{}".format(pre, acc, rec, f1))

