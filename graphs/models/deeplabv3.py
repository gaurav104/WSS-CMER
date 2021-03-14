import torch
import torch.nn as nn
import torch.nn.functional as F

from .assp import ASSP
from .resnet_50 import ResNet_50

class DeepLabv3(nn.Module):
  
	def __init__(self, in_channels, out_channels):
	
		super(DeepLabv3, self).__init__()
		
		self.resnet = ResNet_50(in_channels,64)
		
		self.assp = ASSP(in_channels = 1024)
		self.backbone = self.assp
		
		self.classifier = nn.Conv2d(in_channels = 256, out_channels = out_channels,
							  kernel_size = 1, stride=1, padding=0)
		self.gap = nn.AdaptiveAvgPool2d((1,1))
		

	def forward(self,x):
		_, _, h, w = x.shape
		x = self.resnet(x)
		x = self.assp(x)
		x = self.classifier(x)
		cam = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) #scale_factor = 16, mode='bilinear')
		logits = self.gap(cam)
		return logits, cam
