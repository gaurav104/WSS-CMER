"""
This file will contain the metrics of the framework
"""
import numpy as np
import torch    
from kornia.utils import one_hot
import torch.nn.functional as F

from typing import Optional

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

# Metric helpers


class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure like mean_iou_per_class
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


def cls_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res

def IoU(y_pred, y_target, num_cls=2):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    y_target = F.one_hot(y_target, num_cls)
    y_target = y_target.permute(0,3 ,1, 2)
    y_pred = F.softmax(y_pred , dim =1)

    y_pred = (y_pred > 0.5).float() 
    y_target = (y_target>0.5).float()
    


    intersection = torch.sum(y_target * y_pred, (0, 2, 3))
    union = torch.sum(y_target, (0, 2, 3) ) + torch.sum(y_pred, (0, 2, 3)) - intersection
    return (intersection + 1e-6)  / (union + 1e-6)

def Dice(y_pred, y_target, num_cls=2):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    # assert y_true.dtype == bool and y_pred.dtype == bool
    y_target = F.one_hot(y_target, num_cls)
    y_target = y_target.permute(0,3 ,1, 2)
    y_pred = F.softmax(y_pred , dim =1)

    y_pred = (y_pred > 0.5).float() 
    y_target = (y_target>0.5).float()
    


    intersection = torch.sum(y_target * y_pred, (0, 2, 3))
    union = torch.sum(y_target, (0, 2, 3) ) + torch.sum(y_pred, (0, 2, 3)) - intersection
    return (2*intersection + 1e-6)  / (union + intersection + 1e-6)
