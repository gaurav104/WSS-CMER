"""
This file will contain the metrics of the framework
"""
import torch    
import torch.nn as nn

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