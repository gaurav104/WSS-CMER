import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils import one_hot

class SoftDiceLoss(nn.Module):
    '''
    Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.smooth = 1.

    def forward(self, logits, targets):
        
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) /(iflat.sum() + tflat.sum() + self.smooth))


class InvSoftDiceLoss(nn.Module):

    '''
    Inverted Soft Dice Loss
    '''   
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

        self.smooth = 1.

    def forward(self, logits, targets):
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
    
    
        return 1 - ((2. * intersection + self.smooth) /(iflat.sum() + tflat.sum() + self.smooth))


