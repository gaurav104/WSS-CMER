import torch
import torch.nn as nn
import torch.nn.functional as F


class res_conv_input(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_conv_input, self).__init__()
        self.BN_1 = nn.BatchNorm2d(out_ch)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.conv_identity = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.BN_1(x)
        x = self.ReLU(x)
        x = self.conv_2(x)

        skip = self.conv_identity(inputs)

        addition = skip + x

        return addition



class res_double_conv(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(res_double_conv, self).__init__()
        self.BN_1 = nn.BatchNorm2d(in_ch)
        self.BN_2 = nn.BatchNorm2d(out_ch)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv_identity = nn.Conv2d(in_ch, out_ch, 1)


    def forward(self, inputs):
        x = self.BN_1(inputs)
        x = self.ReLU(x)
        x = self.conv_1(x)

        x = self.BN_2(x)
        x = self.ReLU(x)
        x = self.conv_2(x)

        skip = self.conv_identity(inputs)
        addition = skip + x

        return addition


class inconv(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.res_conv_input = res_conv_input(in_ch, out_ch)

    def forward(self, inputs):
        return self.res_conv_input(inputs)


class outconv(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, inputs):
        return self.conv_1(inputs)



class down(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.res_double_conv = res_double_conv(in_ch, out_ch)

    def forward(self, inputs):
        
        x = self.max_pool1(inputs)
        x = self.res_double_conv(x)
        return x

class up(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.res_double_conv = res_double_conv(in_ch, out_ch)

    def forward(self, input_1, input_2):

        input_1= self.up(input_1)
        x = torch.cat((input_1, input_2), axis = 1)
        x = self.res_double_conv(x)
        return x