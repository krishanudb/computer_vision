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
import cv2

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

##Loading the model
print('Loading Model')
model.load_state_dict(torch.load("model_v1.pth"))
model.eval()

##Post-processing
print('Capturing Video')
outfilename = "result2.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(outfilename,fourcc, 30, (640, 480))
bgex = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)
ret=True
while ret:
    ret, frame = cap.read() #Capture each frame
    fframe = frame #Save final frame
    fgmask = bgex.apply(frame)
    frame = cv2.resize(frame, (50, 50))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fgmask = cv2.resize(fgmask, (50, 50)).reshape((50, 50, 1))
    frame = np.concatenate((frame, fgmask), axis = 2)
    frame = frame.reshape(50,50,4,1)
    frame = frame.transpose((3,2,0,1))
    input = torch.from_numpy(frame).float()
    #Input to model
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    print(predicted.item())

    if predicted.item() == 1:
        case = "stop"
    elif predicted.item() == 2:
        case = "next" 
    elif predicted.item() == 3:
        case = "previous"
    elif predicted.item() == 0:
        case = "none"
    print(case)
    cv2.putText(fframe, '%s' %(case),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.imshow("Gesture Detector", fframe)
    out.write(fframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyWindow("Gesture Detector")


