import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import os
import math
import kornia.augmentation as F


"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
	if method == 'cosine':
		T_total = config.max_epoch * nBatch
		T_cur = (epoch % config.max_epoch) * nBatch + batch
		lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
	else:
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr = config.learning_rate * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr



def export_jsondump(writer):

	assert isinstance(writer, torch.utils.tensorboard.SummaryWriter)

	tf_files = [] # -> list of paths from writer.log_dir to all files in that directory
	for root, dirs, files in os.walk(writer.log_dir):
		for file in files:
			tf_files.append(os.path.join(root,file)) # go over every file recursively in the directory

	for file_id, file in enumerate(tf_files):

		path = os.path.join('/'.join(file.split('/')[:-1])) # determine path to folder in which file lies
		name = os.path.join(file.split('/')[-2]) if file_id > 0 else os.path.join('data') # seperate file created by add_scalar from add_scalars

		# print(file, '->', path, '|', name)

		event_acc = event_accumulator.EventAccumulator(file)
		event_acc.Reload()
		data = {}

		hparam_file = False # I save hparam files as 'hparam/xyz_metric'
		for tag in sorted(event_acc.Tags()["scalars"]):
			if tag.split('/')[0] == 'hparam': hparam_file=True # check if its a hparam file
			step, value = [], []

			for scalar_event in event_acc.Scalars(tag):
				step.append(scalar_event.step)
				value.append(scalar_event.value)

			data[tag] = (step, value)

		if not hparam_file and bool(data):
			with open(path+f'/{name}.json', "w") as f:
				json.dump(data, f)

		 # if its not a hparam file and there is something in the data -> dump it
			

"""
Transforms
"""
# class Flipping(nn.Module):
# 	def __init__(self):
# 		super(SegAugPipeline, self).__init__()
# 		self.denormalize = F.Denormalize(0., 255.)
# 		self.rand_horizontal = F.RandomHorizontalFlip(align_corners=True)
# 		self.rand_vertical = F.RandomVerticalFlip(align_corners=True)
# 		# self.affine = F.RandomAffine(degrees=0,translate=(0.3,0.3))
# 		self.normalize = F.Normalize(0., 255.)

# 	def forward(self, x, y):

# 		horizonatal_param = self.rand_horizontal.generate_parameters(x.shape)
# 		vertical_param = self.rand_vertical.generate_parameters(x.shape)

# 		y = self.denormalize(y)
# 		x = self.denormalize(x)

# 		y = self.rand_horizontal(y, horizonatal_param)
# 		x = self.rand_horizontal(x, horizonatal_param)
		
# 		y = self.rand_vertical(y, vertical_param)
# 		x = self.rand_vertical(x, vertical_param)
		

# 		y = self.normalize(y.round())
# 		x = self.normalize(x.round())

		

# 		return x, y

# class Rotation(nn.Module):
# 	def __init__(self):
# 		super(SegAugPipeline, self).__init__()
# 		self.denormalize = F.Denormalize(0., 255.)
# 		# self.rand_horizontal = F.RandomHorizontalFlip(align_corners=True)
# 		# self.rand_vertical = F.RandomVerticalFlip(align_corners=True)

# 		self.rand_rotate = F.RandomRotation((-180, 180), align_corners=True)
# 		# self.affine = F.RandomAffine(degrees=0,translate=(0.3,0.3))
# 		self.normalize = F.Normalize(0., 255.)

# 	def forward(self, x, y):

# 		# horizonatal_param = self.rand_horizontal.generate_parameters(x.shape)
# 		# vertical_param = self.rand_vertical.generate_parameters(x.shape)
# 		rotation_param = self.rand_rotate.generate_parameters(x.shape)

# 		y = self.denormalize(y)
# 		x = self.denormalize(x)

# 		y = self.rand_rotate(y, rotation_param)
# 		x = self.rand_rotate(x, rotation_param)

# 		y = self.normalize(y.round())
# 		x = self.normalize(x.round())

		
# 		return x, y

# class Scaling(nn.Module):
# 	def __init__(self):
# 		super(SegAugPipeline, self).__init__()
# 		self.denormalize = F.Denormalize(0., 255.)
# 		# self.rand_horizontal = F.RandomHorizontalFlip(align_corners=True)
# 		# self.rand_vertical = F.RandomVerticalFlip(align_corners=True)

# 		self.rand_scale = F.RandomAffine(degrees=0, scale = (0.2, 0.7), align_corners=True)
# 		# self.affine = F.RandomAffine(degrees=0,translate=(0.3,0.3))
# 		self.normalize = F.Normalize(0., 255.)

# 	def forward(self, x, y):

# 		# horizonatal_param = self.rand_horizontal.generate_parameters(x.shape)
# 		# vertical_param = self.rand_vertical.generate_parameters(x.shape)
# 		scale_param = self.rand_scale.generate_parameters(x.shape)

# 		y = self.denormalize(y)
# 		x = self.denormalize(x)

# 		y = self.rand_scale(y, scale_param)
# 		x = self.rand_scale(x, scale_param)

# 		y = self.normalize(y.round())
# 		x = self.normalize(x.round())

		
# 		return x, y

# class Translation(nn.Module):
# 	def __init__(self):
# 		super(SegAugPipeline, self).__init__()
# 		self.denormalize = F.Denormalize(0., 255.)
# 		# self.rand_horizontal = F.RandomHorizontalFlip(align_corners=True)
# 		# self.rand_vertical = F.RandomVerticalFlip(align_corners=True)

# 		self.rand_translate = F.RandomAffine(degrees=0, translate = (0.4, 0.4), align_corners=True)
# 		# self.affine = F.RandomAffine(degrees=0,translate=(0.3,0.3))
# 		self.normalize = F.Normalize(0., 255.)

# 	def forward(self, x, y):

# 		# horizonatal_param = self.rand_horizontal.generate_parameters(x.shape)
# 		# vertical_param = self.rand_vertical.generate_parameters(x.shape)
# 		translate_param = self.rand_translate.generate_parameters(x.shape)

# 		y = self.denormalize(y)
# 		x = self.denormalize(x)

# 		y = self.rand_translate(y, scale_param)
# 		x = self.rand_translate(x, scale_param)

# 		y = self.normalize(y.round())
# 		x = self.normalize(x.round())

		
# 		return x, y




# class AllTransorms(nn.Module):
# 	def __init__(self,):
# 		super(SegAugPipeline, self).__init__()
# 		self.denormalize = F.Denormalize(0., 255.)
# 		self.rand_horizontal = F.RandomHorizontalFlip(align_corners=True)
# 		self.rand_vertical = F.RandomVerticalFlip(align_corners=True)
# 		self.affine = F.RandomAffine(degrees=(-180, 180),translate=(0.4,0.4),scale= (0.2, 0.7), align_corners=True)
# 		self.normalize = F.Normalize(0., 255.)

# 		# F.RandomAffine(degrees=30,translate=(0.3,0.3))
# 	def forward(self, x, y):
# 		# mask_tensor = mask_tensor.unsqueeze(1)
# 		horizonatal_param = self.rand_horizontal.generate_parameters(x.shape)
# 		vertical_param = self.rand_vertical.generate_parameters(x.shape)
# 		affine_param = self.affine.generate_parameters(x.shape)
		
# 		y = self.denormalize(y)
# 		x = self.denormalize(x)

# 		y = self.rand_horizontal(y, horizonatal_param)
# 		x = self.rand_horizontal(x, horizonatal_param)
		
# 		y = self.rand_vertical(y, vertical_param)
# 		x = self.rand_vertical(x, vertical_param)
		
# 		y = self.affine(y, affine_param)
# 		x = self.affine(x, affine_param)
		
# 		y = self.normalize(y.round())
# 		x = self.normalize(x.round())
		
# 		# mask_tensor = mask_tensor.squeeze(1)
# 		# mask_tensor = mask_tensor.long()
		

# 		return x, y