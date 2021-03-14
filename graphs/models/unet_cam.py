import torch
import torch.nn as nn
from .unet_parts import *
from graphs.weights_initializer import init_model_weights

class UNetCAM(nn.Module):
    def __init__(self, n_channels, n_classes,downsize_nb_filters_factor = 4, drop=False):
        super(UNetCAM, self).__init__()
        self.drop = drop
        self.dropout = nn.Dropout2d(p=0.5)
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        # self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        # self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up1 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, bilinear = True)
        self.up2 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor, bilinear = True)
        self.up3 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor, bilinear = True)
        self.backbone = nn.Sequential(nn.Conv2d(64 // downsize_nb_filters_factor,64 // downsize_nb_filters_factor, kernel_size=3, padding =1),
            nn.BatchNorm2d(64 // downsize_nb_filters_factor), 
            nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Conv2d(in_channels=64 // downsize_nb_filters_factor, out_channels=n_classes, kernel_size=(1,1), bias=False)
        # self.outc = outconv(64 // downsize_nb_filters_factor, n_classes)
        # self.Dropout = nn.Dropout(0.5)
        self.apply(init_model_weights)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x3 = self.Dropout(x3)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = torch.cat([inp, x], dim=1)
        # x = self.outc(x)
        x = self.backbone(x)

        if self.drop:
            x = self.dropout(x)

        cam = self.classifier(x)
        logits = self.gap(cam)

        return logits, cam


class UNetCAM_large(nn.Module):
    def __init__(self, n_channels, n_classes,downsize_nb_filters_factor = 4, drop=False):
        super(UNetCAM_large, self).__init__()
        self.drop = drop
        self.dropout = nn.Dropout2d(p=0.5)
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor, bilinear = True)
        self.up3 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor, bilinear = True)
        self.up4 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor, bilinear = True)
        self.backbone = nn.Sequential(nn.Conv2d(64 // downsize_nb_filters_factor,64 // downsize_nb_filters_factor, kernel_size=3, padding =1),
            nn.BatchNorm2d(64 // downsize_nb_filters_factor), 
            nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Conv2d(in_channels=64 // downsize_nb_filters_factor, out_channels=n_classes, kernel_size=(1,1), bias=False)
        # self.outc = outconv(64 // downsize_nb_filters_factor, n_classes)
        # self.Dropout = nn.Dropout(0.5)
        self.apply(init_model_weights)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.backbone(x)

        if self.drop:
            x = self.dropout(x)

        cam = self.classifier(x)
        logits = self.gap(cam)

        return logits, cam
    
    


# a = torch.rand((1,3, 240 ,240))
# net = UNetCAM(3, 2)

# logits , cam = net(a)

# print(logits.size())
# print(cam.size())