"""
ResNet50
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from graphs.weights_initializer import init_model_weights

# from ..weights_initializer import weights_init


class ResNet50(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet50,self).__init__()
        
        resnet50_model = models.resnet50(pretrained=True)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone = nn.Sequential(*list(resnet50_model.children())[1:8])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=(1,1), bias=False)


    def forward(self,x):
        N, C, H, W = x.size()
        x = self.input(x)     
        x = self.backbone(x)
        cam = self.classifier(x)
        cam = F.interpolate(cam, size = (H,W), mode='bilinear', align_corners=True)
        logits = self.gap(cam)
       
        
        return logits, cam 