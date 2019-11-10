import numpy as np
import cv2
import matplotlib.pyplot as plt
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
from PIL import Image

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


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

model = CNN_Model(4, 4, 0.3, 0.4)


model.load_state_dict(torch.load("hand_gesture_v2", map_location = torch.device('cpu')))

model.eval()

def capture():
	txt = "None"
	num = 0
	x = cv2.VideoCapture(0)
	ret = True
	bgex = cv2.createBackgroundSubtractorMOG2()

	while(ret):
		num += 1
		ret, frame = x.read()
		frame = frame[40:440, 120:520, :]
		image = frame.copy()
		fgmask = bgex.apply(frame)
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (600, 600))
		fgmask = cv2.resize(fgmask, (400, 400))
		k = cv2.waitKey(10)

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (50, 50))
		fgmask = cv2.resize(fgmask, (50, 50))

		npframe = np.array(frame)[:, :, ::-1].reshape(1, 50, 50, 3)
		npfgmask = np.array(fgmask).reshape(1, 50, 50, 1)
		# final =  npframe.copy()
		final = np.concatenate([npframe, npfgmask], axis = -1)

		if k == 27:
			x.release()
			cv2.destroyAllWindows()
			break
		if num >= 30 and int(num) % 3 == 0:
			txt = run_model(final)
			# print(txt)

		cv2.putText(image, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
		cv2.imshow("image", image)
def run_model(x):
	global model

	x = x.transpose(0, 3, 1, 2)
	x = torch.from_numpy(x)
	x = x.type(torch.FloatTensor)

	y = model(x)
	_, predicted = torch.max(y.data, 1)
	if predicted.item() == 0:
		return "None"
	elif predicted.item() == 1:
		return "Stop"
	elif predicted.item() == 2:
		return "Play"
	elif predicted.item() == 3:
		return "Back"
capture()