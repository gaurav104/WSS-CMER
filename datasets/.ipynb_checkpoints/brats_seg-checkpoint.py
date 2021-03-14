import os
import glob

import numpy as np
# import scipy.io as sio
from PIL import Image
import kornia.augmentation as F


import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import subprocess
from random import random

class SegAugPipeline(nn.Module):
	def __init__(self,):
		super(SegAugPipeline, self).__init__()
		self.denormalize = F.Denormalize(0., 255.)
		self.rand_horizontal = F.RandomHorizontalFlip()
		self.rand_vertical = F.RandomVerticalFlip()
		self.affine = F.RandomAffine(degrees=0,translate=(0.3,0.3))
		self.normalize = F.Normalize(0., 255.)

		# F.RandomAffine(degrees=30,translate=(0.3,0.3))
	def forward(self, input_tensor, mask_tensor):
		mask_tensor = mask_tensor.unsqueeze(1)
		horizonatal_param = self.rand_horizontal.generate_parameters(input_tensor.shape)
		vertical_param = self.rand_vertical.generate_parameters(input_tensor.shape)
		affine_param = self.affine.generate_parameters(input_tensor.shape)
		
		mask_tensor = self.denormalize(mask_tensor)
		input_tensor = self.denormalize(input_tensor)

		mask_tensor = self.rand_horizontal(mask_tensor, horizonatal_param)
		input_tensor = self.rand_horizontal(input_tensor, horizonatal_param)
		
		mask_tensor = self.rand_vertical(mask_tensor, vertical_param)
		input_tensor = self.rand_vertical(input_tensor, vertical_param)
		
		mask_tensor = self.affine(mask_tensor, affine_param)
		input_tensor = self.affine(input_tensor, affine_param)
		
		mask_tensor = self.normalize(mask_tensor)
		input_tensor = self.normalize(input_tensor.round())
		
		mask_tensor = mask_tensor.squeeze(1)
		mask_tensor = mask_tensor.long()
		

		return input_tensor, mask_tensor

class brats(data.Dataset):
	def __init__(self, mode, data_root, modality, transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.modality = modality
		# self.mapping = {
		#     0: 0,
		#     255: 1              
		# }

		self.images_path = data_root + '{}/MR_{}/'.format(mode, modality[0])
		self.masks_path = data_root + '{}/GT_CAM_{}/'.format(mode, modality[0])
		self.images_dirs = glob.glob(self.images_path + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

		# if mode == 'val':
		#     self.images_path = data_root + 'test_image_{}/*.png'.format(self.fold)
		#     self.mask_path = data_root + 'test_mask_{}/*.png'.format(self.fold)
		#     self.images = glob.glob(self.images_path)
		#     self.masks = glob.glob(self.mask_path)
			
		self.images_dirs.sort()
		self.masks_dirs.sort()
	
	# def mask_to_class(self, mask):
	#     for k in self.mapping:
	#         mask[mask==k] = self.mapping[k]
	#     return mask

	def __getitem__(self, index):
		thresh = 127.5
		fn = lambda x : 1 if x > thresh else 0
		image = Image.open(self.images_dirs[index]).convert('L')
		mask = Image.open(self.masks_dirs[index]).convert('L')
		mask = mask.point(fn, mode='1')
		mask = torch.from_numpy(np.asarray(mask,dtype='uint8')).long()
		# mask_array = np.array(mask)



		# mask = torch.from_numpy(mask_array).long() # this is for my dataset(lv)

		if self.transforms is not None:
			image = self.transforms(image)

#             mask = self.transforms(mask)

#         print(np.unique(mask.numpy()))
		return (image, mask)
	
	

	def __len__(self):
		return len(self.masks_dirs)

class brats_2modality(data.Dataset):
	def __init__(self, mode, data_root, modality=['T1','T2'], transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms



		self.images_path_1 = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.images_path_2 = data_root + '{}/MR_{}/'.format(mode ,modality[1])
		self.masks_path = data_root + '{}/GT/'.format(mode, modality[0],modality[1])
		self.images_dirs_1 = glob.glob(self.images_path_1 + "*.png")
		self.images_dirs_2 = glob.glob(self.images_path_2 + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

			
		self.images_dirs_1.sort()
		self.images_dirs_2.sort()
		self.masks_dirs.sort()


	def __getitem__(self, index):
		thresh = 127.5
		fn = lambda x : 1 if x > thresh else 0
		image_1 = Image.open(self.images_dirs_1[index]).convert('L')
		image_2 = Image.open(self.images_dirs_2[index]).convert('L')
		mask = Image.open(self.masks_dirs[index]).convert('L')
		mask = mask.point(fn, mode='1')
		mask = torch.from_numpy(np.asarray(mask,dtype='uint8')).long()



		if self.transforms is not None:
			image_1 = self.transforms(image_1)
			image_2 = self.transforms(image_2)

		image = torch.cat([image_1, image_2], dim=0)

		return (image, mask)
	

	def __len__(self):
		return len(self.masks_dirs)

class BratsLoader:
	def __init__(self, config):
		self.config = config
		
		#IF training on cluster change data root directory
		if self.config.run_on_cluster:
			output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
			output_string = output_bytes.decode('utf-8').strip()
			root = os.path.join(output_string, 'brats_data/')
			self.config.data_root = root

		assert self.config.mode in ['train', 'test', 'random']

		self.config.modality = str(self.config.modality).split(",")

		self.input_transform = transforms.Compose([transforms.ToTensor()])
		

		if self.config.mode == 'train':

			if len(self.config.modality)==1:
				train_set = brats('train', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
				validation_set = brats('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
			elif len(self.config.modality)==2:

				train_set= brats_2modality('train', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
				validation_set = brats_2modality('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
			# test_set = brats('train', self.config.data_root,
			#                     modality =self.config.modality,
			#                     transforms=self.input_transform_test                              
			#                    )

			self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
			self.validation_loader = DataLoader(validation_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
			# self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
			#                                num_workers=self.config.data_loader_workers,
			#                                pin_memory=self.config.pin_memory)

			self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
			self.validation_iterations  = (len(validation_set) + self.config.batch_size) // self.config.batch_size
			# self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size
			
		# elif self.config.mode == 'test':
		#     valid_set = CellSeg('test', self.config.data_root,
		#                         fold =self.config.fold,
		#                         transforms=self.input_transform
		#                        )
		#     self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
		#                                    num_workers=self.config.data_loader_workers,
		#                                    pin_memory=self.config.pin_memory)
		#     self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size
			

		# elif self.config.mode == 'test':
		#     test_set = VOC('test', self.config.data_root,
		#                    transform=self.input_transform, target_transform=self.target_transform)

		#     self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
		#                                   num_workers=self.config.data_loader_workers,
		#                                   pin_memory=self.config.pin_memory)
		#     self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

		else:
			raise Exception('Please choose a proper mode for data loading')

	def finalize(self):
		pass

# def classification_augmentations(x):
# 	transforms = 

# 	return transforms(x)





	
if __name__ == '__main__':

	train_set = brats('val', "../../brats_data/",['T1'], transforms.Compose([transforms.ToTensor()]))
	# directory = glob.glob("../../brats_data/train/GT/*.png")
	# print(directory)
	# print(len(directory))

	# # # valid_set = CellSeg('val', "../../Dataset/",
	# # #                            transforms.Compose([transforms.ToTensor()]))
	# # # valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

	train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

	for image, mask in train_loader:
# 		print(mask.dtype)
		aug = SegAugPipeline()
		image, mask= aug(image, mask)
		print(image.size(), mask.size())
		
	#   