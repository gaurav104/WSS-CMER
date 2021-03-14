import torch
import torch.nn as nn
from torchvision import models

class ResNet_50 (nn.Module):
	def __init__(self, in_channels = 1, conv1_out = 64):
		super(ResNet_50,self).__init__()
		self.resnet_50 = models.resnet50(pretrained = True)
		self.resnet_50.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.relu = nn.ReLU(inplace=True)
  
	def forward(self,x):
		x = self.relu(self.resnet_50.bn1(self.resnet_50.conv1(x)))
		x = self.resnet_50.maxpool(x)
		x = self.resnet_50.layer1(x)
		x = self.resnet_50.layer2(x)
		x = self.resnet_50.layer3(x)
		
		return x
