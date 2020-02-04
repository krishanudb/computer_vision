from __future__ import print_function


import numpy as np
import torch
import torchvision
import random
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
import torchvision.transforms.functional as TF

import os
import cv2 as cv
from PIL import Image
from scipy import ndimage

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


loaded = np.load('training_data.npz')


X = loaded["inp"]
Y = loaded["out"].reshape(-1, 1)


class MyDataset(Dataset):
    def __init__(self, data, target, image_transform = None, other_transform = None):
        self.data = data
        self.target = target
        self.img_transform = image_transform
        self.oth_transform = other_transform
        
    def __getitem__(self, index):
        x = self.data[index][:, :, :3]
#         x = self.data[index][:, :, :3]
        x = x[:, :, ::-1]
        b = self.data[index][:, :, 3]
        y = self.target[index]
        
        x = Image.fromarray(x.astype('uint8'), 'RGB')        
        b = Image.fromarray(b.astype('uint8'), 'L')

        sample = {"x": x, "y": y}
        sample = {"x": x, "b": b, "y": y}
        if self.img_transform:
            sample["x"] = self.img_transform(sample["x"])
            sample["b"] = self.img_transform(sample["b"])
        if self.oth_transform:
            sample = self.oth_transform(sample)
        return sample
    
    def __len__(self):
        return self.data.shape[0]
    
class ToTensor(object):
    
    def __call__(self, sample):
        x = sample["x"]
        b = sample["b"]
#         x = np.array(sample['x']).astype(np.uint8)
#         b = np.array(sample['b']).astype(np.uint8)
        
        # Rotate:
        if random.random() < 0.80:
            angle = random.randint(-30, 30)
            x = TF.rotate(x, angle)
            b = TF.rotate(b, angle) 
        # Flip
        if random.random() < 0.40:
            x = TF.vflip(x)
            b = TF.vflip(b)
        x = np.array(x) / 255.
        b = np.array(b).reshape(50, 50, 1) / 255.
        x = np.concatenate([x, b], axis = -1)
        x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        
        y = sample["y"]
        y = np.long(y[0])
        
        return {"x": x, "y": y}

    
# Random Transforms

jitter = torchvision.transforms.ColorJitter(.250, .250, .250, .250)
# rotate = torchvision.transforms.RandomRotation(30)
# flip = torchvision.transforms.RandomVerticalFlip(p=0.2)
hand_dataset = MyDataset(X, Y, image_transform = transforms.Compose([jitter]), other_transform = transforms.Compose([ToTensor()]))


train_size = 16000
val_size = 400
test_size = len(hand_dataset) - train_size - val_size
train_data, val_data, test_data= random_split(hand_dataset, [train_size, val_size, test_size])

# print("training")
# for i in range(len(train_data)):
#     sample = train_data[i]
#     print(i, sample['x'].size())
#     if i == 5:
#         break

# print("validation")
# for i in range(len(val_data)):
#     sample = val_data[i]
#     print(i, sample['x'].size())
#     if i == 2:
#         break


train_loader = DataLoader(dataset = train_data, batch_size = 400, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset= val_data, batch_size = 400, shuffle=True, num_workers=0)

classes = ('0','1','2','3')
class CNN_Model(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate, dropout_rate_fc):
        super(CNN_Model, self).__init__()
    
        self.down1 = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=3, padding = 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout_rate))
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding = 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout_rate))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
        
        self.down3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding = 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout_rate))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.ff1 = nn.Sequential(nn.Linear(128 * 7 * 7, 64),  
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Dropout2d(dropout_rate_fc))
        
        self.output = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.down1(x)
        x = self.pool1(x)
        x = self.down2(x)
        x = self.pool2(x)
        x = self.down3(x)
        x = self.pool3(x)
        x = x.reshape(-1, 128 * 7 * 7)
        x = self.ff1(x)
        x = self.output(x)
        return x

model = CNN_Model(4, 4, 0.2, 0.5)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

import time

epochs = 2

loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    model = model.cuda()


print("Training Started")
for epoch in range(epochs):
    start_time = time.time() 
    
    loss_temp = []
    acc_temp = []
    
    for mini_batch_num, data in enumerate(train_loader):
        images, labels = data['x'], data['y']
#         print(labels)
        images, labels = images.to(device), labels.to(device)
        model.train()
        preds = model(images)
        loss = criterion(preds, labels)

        _, predicted = torch.max(preds.data, 1)
        accuracy = torch.mean((predicted == labels).type(torch.FloatTensor)).item()
        optimizer.zero_grad()
        loss.backward()
        loss_temp.append(loss.item())
        acc_temp.append(accuracy)
        optimizer.step()
        
        if (mini_batch_num) % 5 == 0:
            print ('Epoch {}/{}; Iter {}/{}; Loss: {:.4f}; Acc: {}'.format(epoch+1, epochs, mini_batch_num + 1, len(train_loader), loss.item(), accuracy), end = "\r")
    
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            model.eval()
            images, labels = data['x'], data['y']
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss_test_temp = criterion(preds, labels)
            _, predicted = torch.max(preds.data, 1)
            accuracy = torch.mean((predicted == labels).type(torch.FloatTensor)).item()
        end_time = time.time()
        print ('Epoch {}/{}; Iter {}/{}; Training Loss: {:.4f}; Test Loss: {}; Training Acc: {}; Validation Accuracy {}'.format(epoch+1, epochs, mini_batch_num + 1, \
                len(train_loader), round(np.mean(np.array(loss_temp)), 3), round(loss_test_temp.item(), 3), round(np.mean(np.array(acc_temp)), 3), round(accuracy, 3)))
    
    loss_train.append(np.mean(np.array(loss_temp)))
    accuracy_train.append(np.mean(np.array(acc_temp)))
    loss_test.append(loss_test_temp.item())
    accuracy_test.append(accuracy)

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))
plt.plot(loss_train, c="red", label="Training Loss")
# plt.legend("Training Loss")
plt.plot(loss_test, c="green", label = "Validation Loss")
# plt.legend("Test Loss")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show();

plt.figure(figsize = (10, 6))
plt.plot(accuracy_train, c="red", label="Training Accuracy")
# plt.legend("Training Loss")
plt.plot(accuracy_test, c="green", label = "Validation Accuracy")
# plt.legend("Test Loss")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show();

torch.save(model.state_dict(), "hand_gesture_model")