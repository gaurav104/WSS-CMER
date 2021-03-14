import torch
import torch.nn as nn
import torch.nn.functional as F

class attention(nn.Module):
    '''Channel Attention'''
    def __init__(self, in_ch, out_ch):
        super(attention, self).__init__()

        self.conv_gen = nn.Conv2d(in_ch, in_ch, 3, padding = 1)
        self.conv_attn = nn.Conv2d(in_ch, in_ch, 3, padding = 1)
        self.conv_output = nn.Conv2d(1, out_ch, 1, padding  =0 )


    def forward(self, x):
        g = self.conv_gen(x)
        g = torch.tanh(g)

        attn = self.conv_attn(x)
        attn = F.softmax(attn, 1)

        Ig = torch.sum(torch.mul(g, attn), dim= 1, keepdim = True)

        out = self.conv_output(Ig)
        return out