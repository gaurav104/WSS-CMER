import os
import glob

import numpy as np
from PIL import Image
import kornia.augmentation as F

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import subprocess
import random
import kornia as K

from PIL import Image, ImageOps
from random import random, randint, uniform, choice


class Rotate90(F.AugmentationBase):

	def __init__(self, return_transform = False):
		super(Rotate90, self).__init__()

	def generate_parameters(self, input_shape):

		angles_rad = torch.randint(1, 4, (input_shape[0],)) * K.pi/2.
		angles_deg = K.rad2deg(angles_rad)
		return dict(angles=angles_deg)

	def compute_transformation(self, input, params):
	  # compute transformation
		B, C ,H ,W = input.shape
		angles = params['angles'].type_as(input)
		center = torch.tensor([[W / 2, H / 2]]*B).type_as(input)
		transform = K.get_rotation_matrix2d(
		center, angles, torch.ones_like(angles))
		return transform

	def apply_transform(self, input, params):

		# compute transformation
		B, C, H, W = input.shape
		transform = self.compute_transformation(input, params)
		# apply transformation and return
		output = K.warp_affine(input, transform, (H, W))
		return output

class All(nn.Module):
	def __init__(self):
		super(All, self).__init__()
		self.transform1 = Flip()
		self.transform2 = Rotate()
		self.transform3 = Translate()
		self.transform4 = Resize()
		self.transforms = [self.transform1, self.transform2, self.transform3, self.transform4]

	def forward(self, input_tensor, other_tensor=None):
		apply_transform = choice(self.transforms)
		if other_tensor is not None:
			input_tensor, other_tensor = apply_transform(input_tensor, other_tensor)
			return input_tensor, other_tensor
		return apply_transform(input_tensor)

class Flip(nn.Module):
	def __init__(self):
		super(Flip, self).__init__()
		self.transform_hflip = F.RandomHorizontalFlip(p=1.0, align_corners=True)
		self.transform_vflip = F.RandomVerticalFlip(p=1.0)

	def forward(self, input_tensor, other_tensor=None):
		k = random()
		if k >= 0.5:
			self.transform = self.transform_hflip
			transformation_param = self.transform_hflip.generate_parameters(input_tensor.shape)
		else:
			self.transform = self.transform_vflip
			transformation_param = self.transform_vflip.generate_parameters(input_tensor.shape)

		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor
		return input_tensor

class Rotate(nn.Module):
	def __init__(self):
		super(Rotate, self).__init__()
		self.transform = Rotate90()

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 
		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor
		return input_tensor

class Translate(nn.Module):
	def __init__(self):
		super(Translate, self).__init__()
		self.transform = F.RandomAffine(degrees=0, translate=(0.3,0.3), align_corners=True)

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 
		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor

		return input_tensor


class Resize(nn.Module):
	def __init__(self):
		super(Resize, self).__init__()
		self.sizes = [i for i in range(192,296, 8)]

	def forward(self, input_tensor, other_tensor=None):
		output_size  = choice(self.sizes)
		input_tensor = nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners= True, recompute_scale_factor=True)
		if other_tensor is not None:
			other_tensor = nn.functional.interpolate(other_tensor, size=output_size, mode='bilinear', align_corners= True, recompute_scale_factor=True)
			return input_tensor, other_tensor

		return input_tensor

class brats(data.Dataset):
	def __init__(self, mode, data_root, modality, augmentation = False, transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.modality = modality
		self.augmentation = augmentation

		self.images_path = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.images_dirs = glob.glob(self.images_path + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")
		self.image_name_list = os.listdir(self.images_path)

		self.images_dirs.sort()
		self.masks_dirs.sort()
		self.image_name_list.sort()

		self.items = []

		for image_dir, mask_dir, image_name in zip(self.images_dirs, self.masks_dirs, self.image_name_list):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0

			item = (image_dir, label, image_name)
			self.items.append(item)

	def augment(self, img):
		# if random() > 0.5:
		# 	img = ImageOps.flip(img)
		# 	# mask = ImageOps.flip(mask)
		# if random() > 0.5:
		# 	img = ImageOps.mirror(img)
		# 	# mask = ImageOps.mirror(mask)
		# if random() > 0.5:
		# 	angle = random() * 60 - 30
		# 	img = img.rotate(angle)
		# 	mask = mask.rotate(angle)
		# if random() > 0.5:
		# 	width, height = img.size
		# 	x_off = random.uniform(-width*0.1, width*0.1)
		# 	y_off = random.uniform(-height*0.1, height*0.1)
		# 	img = Image.offset(img, int(x_off), int(y_off))

		return img

	def __getitem__(self, index):

		image_dir, label, image_name = self.items[index]
		image = Image.open(image_dir).convert('L')
		label = torch.squeeze(torch.Tensor([label])).long()
		
		if self.augmentation:
			image = self.augment(image)

		if self.transforms is not None:
			image = self.transforms(image)
		return (image, label, image_name)
	
	def __len__(self):
		return len(self.masks_dirs)



class brats_2modality(data.Dataset):
	def __init__(self, mode, data_root, modality=['T1','T2'], augmentation = False,transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.augmentation = augmentation

		self.images_path_1 = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.images_path_2 = data_root + '{}/MR_{}/'.format(mode ,modality[1])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.image_name_list = os.listdir(self.masks_path)

		self.images_dirs_1 = glob.glob(self.images_path_1 + "*.png")
		self.images_dirs_2 = glob.glob(self.images_path_2 + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

		self.images_dirs_1.sort()
		self.images_dirs_2.sort()
		self.masks_dirs.sort()
		self.image_name_list.sort()

		self.items = []
		for image_dir_1, image_dir_2, mask_dir, image_name in zip(self.images_dirs_1, self.images_dirs_2, self.masks_dirs, self.image_name_list):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0
			item = (image_dir_1, image_dir_2, label, image_name)
			self.items.append(item)

	def augment(self, img1, img2):
		
		# if random() > 0.5:
		# 	img1 = ImageOps.flip(img1)
		# 	img2 = ImageOps.flip(img2)
		# 	mask = ImageOps.flip(mask)
		# if random() > 0.5:
		# 	img1 = ImageOps.mirror(img1)
		# 	img2 = ImageOps.mirror(img2)
		# 	mask = ImageOps.mirror(mask)
		# if random() > 0.5:
		# 	angle = random() * 60 - 30
		# 	img1 = img1.rotate(angle)
		# 	img2 = img2.rotate(angle)
		# 	mask = mask.rotate(angle)
		# if random() > 0.5:
		# 	width, height = img.size
		# 	x_off = random.uniform(-width*0.1, width*0.1)
		# 	y_off = random.uniform(-height*0.1, height*0.1)
		# 	img = Image.offset(img, int(x_off), int(y_off))
		return img1, img2


	def __getitem__(self, index):

		image_dir_1, image_dir_2, label, image_name = self.items[index]
		
		image_1 = Image.open(image_dir_1).convert('L')
		image_2 = Image.open(image_dir_2).convert('L')

		label = torch.squeeze(torch.Tensor([label])).long()
		if self.augmentation:
			image_1, image_2 = self.augment(image_1, image_2)

		if self.transforms is not None:
			image_1 = self.transforms(image_1)
			image_2 = self.transforms(image_2)

		image_tensor = torch.cat([image_1, image_2], dim=0)
		return (image_tensor, label, image_name)
	
	def __len__(self):
		return len(self.masks_dirs)


class brats_4modality(data.Dataset):
	def __init__(self, mode, data_root, modality=['T1','T2', 'T1c', 'Flair'], augmentation = False, transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.augmentation = augmentation

		self.images_path_1 = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.images_path_2 = data_root + '{}/MR_{}/'.format(mode ,modality[1])
		self.images_path_3 = data_root + '{}/MR_{}/'.format(mode ,modality[2])
		self.images_path_4 = data_root + '{}/MR_{}/'.format(mode ,modality[3])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.image_name_list = os.listdir(self.masks_path)

		self.images_dirs_1 = glob.glob(self.images_path_1 + "*.png")
		self.images_dirs_2 = glob.glob(self.images_path_2 + "*.png")
		self.images_dirs_3 = glob.glob(self.images_path_3 + "*.png")
		self.images_dirs_4 = glob.glob(self.images_path_4 + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

		self.images_dirs_1.sort()
		self.images_dirs_2.sort()
		self.images_dirs_3.sort()
		self.images_dirs_4.sort()
		self.masks_dirs.sort()
		self.image_name_list.sort()

		self.items = []
		for image_dir_1, image_dir_2, image_dir_3, image_dir_4, mask_dir, image_name in zip(self.images_dirs_1, self.images_dirs_2, self.images_dirs_3, self.images_dirs_4, self.masks_dirs, self.image_name_list):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0

			item = (image_dir_1, image_dir_2, image_dir_3, image_dir_4, label, image_name)
			self.items.append(item)

	def augment(self, img1, img2, img3, img4):
		# if random() > 0.5:
		# 	img1 = ImageOps.flip(img1)
		# 	img2 = ImageOps.flip(img2)
		# 	img3 = ImageOps.flip(img3)
		# 	img4 = ImageOps.flip(img4)
		# 	mask = ImageOps.flip(mask)
		# if random() > 0.5:
		# 	img1 = ImageOps.mirror(img1)
		# 	img2 = ImageOps.mirror(img2)
		# 	img3 = ImageOps.mirror(img3)
		# 	img4 = ImageOps.mirror(img4)
		# 	mask = ImageOps.mirror(mask)
		# if random() > 0.5:
		# 	angle = random() * 60 - 30
		# 	img1 = img1.rotate(angle)
		# 	img2 = img2.rotate(angle)
		# 	img3 = img3.rotate(angle)
		# 	img4 = img4.rotate(angle)
		# 	mask = mask.rotate(angle)
		return img1, img2, img3, img4

	def __getitem__(self, index):

		image_dir_1, image_dir_2, image_dir_3, image_dir_4, label, image_name = self.items[index]
		
		image_1 = Image.open(image_dir_1).convert('L')
		image_2 = Image.open(image_dir_2).convert('L')
		image_3 = Image.open(image_dir_3).convert('L')
		image_4 = Image.open(image_dir_4).convert('L')

		label = torch.squeeze(torch.Tensor([label])).long()

		if self.augmentation:
			image_1, image_2, image_3, image_4 = self.augment(image_1, image_2, image_3, image_4)

		if self.transforms is not None:
			image_1 = self.transforms(image_1)
			image_2 = self.transforms(image_2)
			image_3 = self.transforms(image_3)
			image_4 = self.transforms(image_4)

		image_tensor = torch.cat([image_1, image_2, image_3, image_4], dim=0)

		return (image_tensor, label, image_name)
	
	def __len__(self):
		return len(self.masks_dirs)

class BratsLoader:
	def __init__(self, config):
		self.config = config

		if self.config.run_on_cluster:
			output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
			output_string = output_bytes.decode('utf-8').strip()
			root_classification = os.path.join(output_string, 'brats_classification/')
			root_infer = os.path.join(output_string, 'brats_data/')
			self.config.data_root = root_classification
			self.config.data_root_infer = root_infer

		assert self.config.mode in ['train', 'test', 'random']

		self.config.modality = str(self.config.modality).split(",")
   
		self.input_transform_train = transforms.Compose([transforms.transfors.RandomAffine(degrees=0, translate=(0.1,0.1)),
														 transforms.transforms.ToTensor()])
		self.input_transform = transforms.Compose([transforms.ToTensor()])

		if self.config.mode == 'train':

			if len(self.config.modality)==1:
				train_set = brats('train', self.config.data_root,
									modality = self.config.modality,
									augmentation = self.config.augmentation,
									transforms=self.input_transform_train
								   )
				validation_set = brats('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)

				validation_set_infer = brats('val', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
				test_set_infer = brats('test', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
			elif len(self.config.modality)==2:

				train_set= brats_2modality('train', self.config.data_root,
									modality = self.config.modality,
									augmentation = self.config.augmentation,
									transforms=self.input_transform_train
									)
				validation_set = brats_2modality('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform
									)
				validation_set_infer = brats_2modality('val', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
				test_set_infer = brats_2modality('test', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
									)
			elif len(self.config.modality)==4:

				train_set= brats_4modality('train', self.config.data_root,
									modality = self.config.modality,
									augmentation = self.config.augmentation,
									transforms=self.input_transform)
				validation_set = brats_4modality('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
				validation_set_infer = brats_4modality('val', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
				test_set_infer = brats_4modality('test', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform)

			self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
			self.validation_loader = DataLoader(validation_set, batch_size=self.config.batch_size, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)

			self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
			self.validation_iterations  = (len(validation_set) + self.config.batch_size) // self.config.batch_size

			
			self.validation_infer_loader = DataLoader(validation_set_infer, batch_size=1, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
			self.test_infer_loader = DataLoader(test_set_infer, batch_size=1, shuffle=False,
										   num_workers=self.config.data_loader_workers,
										   pin_memory=self.config.pin_memory)
			# self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
			#                                num_workers=self.config.data_loader_workers,
			#                                pin_memory=self.config.pin_memory)

			self.validation_infer_iterations = (len(validation_set_infer) + self.config.batch_size) // self.config.batch_size
			self.test_infer_iterations  = (len(test_set_infer) + self.config.batch_size) // self.config.batch_size


	def finalize(self):
		pass


if __name__ == '__main__':

	train_set = brats_4modality('train', "../../brats_classification/",['T1','T2','T1c','Flair'], transforms=transforms.Compose([transforms.ToTensor()]))
	train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
	for image, label in train_loader:
		print(image.size())
	#     
