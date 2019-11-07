import numpy as np
import torch
import torchvision
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
import os
import cv2 as cv

##Loading the dataset    
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return self.data.shape[0]

hand_dataset = np.load('datasetbg.npz')
images = hand_dataset["image"]
annotations = hand_dataset["ann"]
annotations[annotations == 4] = 0
images = images.transpose((0,3, 1, 2))
hand_dataset = MyDataset(images, annotations)

##Spliting datset
train_size = 15600
val_size = 65
test_size = 50
train_data, val_data, test_data = random_split(hand_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(dataset = train_data, batch_size = 30, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset= val_data, batch_size = 15, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset= test_data, batch_size = 10, shuffle=True, num_workers=0)
classes = ('0','1','2','3')
train_dataiter = iter(train_loader)
images, annotations = train_dataiter.next()
print('inputs size: ', images.shape)
print('labels size: ', annotations.shape)

##Defining the network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc4 = nn.Linear(32 * 50 * 50, 120)
        self.fc5 = nn.Linear(120, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
#         print(x.size())
        x = F.relu(self.fc4(x.view(-1,32*50*50)))
        x = self.fc5(x)
        return x
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

##Defining hyperparameters
batchsize=200 #size of minibatch
Epoch=20 #the number of iteration

##Training the network
print('Started Training')
mini_batch = batchsize
loss_values = []
val_loss_values = []
for epoch in range(Epoch): 
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % mini_batch == mini_batch-1:    # print every 200 mini-batches
            print('[%d, %5d] Training loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / mini_batch))
            loss_values.append(running_loss/mini_batch)
            running_loss = 0.0

    #At the end of the epoch, do a pass on the validation set
    total_val_loss = 0
    for inputs, labels in val_loader:
        inputs, labels = Variable(inputs), Variable(labels)
        val_outputs = model(inputs)
        val_loss_size = criterion(val_outputs, labels)
        total_val_loss += val_loss_size.item()
    print('Validation loss = {:.2f}'.format(total_val_loss / len(val_loader)))
    val_loss_values.append(total_val_loss / len(val_loader))

torch.save(model.state_dict(),'model_v2.pth')

plt.plot(loss_values, label='Training loss')
plt.plot(val_loss_values, label='Validation loss')
plt.legend(frameon=False)
plt.show()
print('Finished Training')


##Chenking Train and Test accuracies
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))