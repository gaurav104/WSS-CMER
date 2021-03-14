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

class ClfAugPipeline(nn.Module):
	def __init__(self,):
		super(ClfAugPipeline, self).__init__()
		self.transforms = nn.Sequential(F.Denormalize(0., 255.),
			F.RandomHorizontalFlip(align_corners=True),
			F.RandomVerticalFlip(),                                        
# 			F.RandomRotation(degrees=15, align_corners=True),                                 
			F.RandomAffine(degrees=0, translate=(0.1,0.1), align_corners=True)
            )
		self.normalize = F.Normalize(0., 255.)
	def forward(self, input_tensor):
		out = self.transforms(input_tensor)
		out = self.normalize(out.round())
		return out

class All(nn.Module):
	def __init__(self):
		super(All, self).__init__()

		self.denormalize = F.Denormalize(0., 255.)
		self.transform1 = F.RandomHorizontalFlip(p=1,align_corners=True)
		self.transform2 = F.RandomRotation(degrees=180, align_corners=True)
		self.transform3 = F.RandomAffine(degrees=0, translate=(0.2,0.2), scale=(0.6,1.2),align_corners=True)
		self.normalize = F.Normalize(0., 255.)

		# self.transform = nn.Sequential(F.Denormalize(0., 255.),
		# 	F.RandomHorizontalFlip(align_corners=True),
		# 	F.Normalize(0., 255.))

	def forward(self, input_tensor, other_tensor=None):

		transformation_param1 = self.transform1.generate_parameters(input_tensor.shape)
		transformation_param2 = self.transform2.generate_parameters(input_tensor.shape)
		transformation_param3 = self.transform3.generate_parameters(input_tensor.shape) 

		input_tensor = self.denormalize(input_tensor)
		input_tensor = self.transform1(input_tensor, transformation_param1)
		input_tensor = self.transform2(input_tensor, transformation_param2)
		input_tensor = self.transform3(input_tensor, transformation_param3)
		input_tensor = self.normalize(input_tensor.round())

		if other_tensor is not None:
			# other_tensor = self.denormalize(other_tensor)
			other_tensor = self.transform1(other_tensor, transformation_param1)
			other_tensor = self.transform2(other_tensor, transformation_param2)
			other_tensor = self.transform3(other_tensor, transformation_param3)
			# other_tensor = self.normalize(other_tensor)
			return input_tensor, other_tensor

		return input_tensor



class HFlip(nn.Module):
	def __init__(self):
		super(HFlip, self).__init__()

		self.denormalize = F.Denormalize(0., 255.)
		self.transform = F.RandomHorizontalFlip(p=1.0, align_corners=True)
		self.normalize = F.Normalize(0., 255.)

		# self.transform = nn.Sequential(F.Denormalize(0., 255.),
		# 	F.RandomHorizontalFlip(align_corners=True),
		# 	F.Normalize(0., 255.))

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 

		input_tensor = self.denormalize(input_tensor)
		input_tensor = self.transform(input_tensor, transformation_param)
		input_tensor = self.normalize(input_tensor.round())

		if other_tensor is not None:
			# other_tensor = self.denormalize(other_tensor)
			other_tensor = self.transform(other_tensor, transformation_param)
			# other_tensor = self.normalize(other_tensor)
			return input_tensor, other_tensor

		return input_tensor

class Rotate(nn.Module):
	def __init__(self):
		super(Rotate, self).__init__()

		self.denormalize = F.Denormalize(0., 255.)
		self.transform = F.RandomRotation(degrees=180, align_corners=True)
		self.normalize = F.Normalize(0., 255.)

		# self.transform = nn.Sequential(F.Denormalize(0., 255.),
		# 	F.RandomHorizontalFlip(align_corners=True),
		# 	F.Normalize(0., 255.))

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 

		input_tensor = self.denormalize(input_tensor)
		input_tensor = self.transform(input_tensor, transformation_param)
		input_tensor = self.normalize(input_tensor.round())

		if other_tensor is not None:
			# other_tensor = self.denormalize(other_tensor)
			other_tensor = self.transform(other_tensor, transformation_param)
			# other_tensor = self.normalize(other_tensor)
			return input_tensor, other_tensor

		return input_tensor

class Translate(nn.Module):
	def __init__(self):
		super(Translate, self).__init__()

		self.denormalize = F.Denormalize(0., 255.)
		self.transform = F.RandomAffine(degrees=0, translate=(0.2,0.2), align_corners=True)
		self.normalize = F.Normalize(0., 255.)

		# self.transform = nn.Sequential(F.Denormalize(0., 255.),
		# 	F.RandomHorizontalFlip(align_corners=True),
		# 	F.Normalize(0., 255.))

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 

		input_tensor = self.denormalize(input_tensor)
		input_tensor = self.transform(input_tensor, transformation_param)
		input_tensor = self.normalize(input_tensor.round())

		if other_tensor is not None:
			# other_tensor = self.denormalize(other_tensor)
			other_tensor = self.transform(other_tensor, transformation_param)
			# other_tensor = self.normalize(other_tensor)
			return input_tensor, other_tensor

		return input_tensor

class Scale(nn.Module):
	def __init__(self):
		super(Scale, self).__init__()

		self.denormalize = F.Denormalize(0., 255.)
		self.transform = F.RandomAffine(degrees=0, scale=(0.6,1.2) ,align_corners=True)
		self.normalize = F.Normalize(0., 255.)

		# self.transform = nn.Sequential(F.Denormalize(0., 255.),
		# 	F.RandomHorizontalFlip(align_corners=True),
		# 	F.Normalize(0., 255.))

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape) 

		input_tensor = self.denormalize(input_tensor)
		input_tensor = self.transform(input_tensor, transformation_param)
		input_tensor = self.normalize(input_tensor.round())

		if other_tensor is not None:
			# other_tensor = self.denormalize(other_tensor)
			other_tensor = self.transform(other_tensor, transformation_param)
			# other_tensor = self.normalize(other_tensor)
			return input_tensor, other_tensor

		return input_tensor




class brats(data.Dataset):
	def __init__(self, mode, data_root, modality, transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.modality = modality
		# self.mapping = {
		#     0: 0,
		#     255: 1              
		# }

		#initilializing imgs with binary labels for classificatio

		self.images_path = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.images_dirs = glob.glob(self.images_path + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

		self.images_dirs.sort()
		self.masks_dirs.sort()

		self.items = []

		for image_dir, mask_dir in zip(self.images_dirs, self.masks_dirs):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0

			item = (image_dir, label)
			self.items.append(item)


	def __getitem__(self, index):

		image_dir, label = self.items[index]
		
		image = Image.open(image_dir).convert('L')
		label = torch.squeeze(torch.Tensor([label])).long()



		# mask = torch.from_numpy(mask_array).long() # this is for my dataset(lv)

		if self.transforms is not None:
			image = self.transforms(image)

#             mask = self.transforms(mask)

#         print(np.unique(mask.numpy()))
		return (image, label)
	

	def __len__(self):
		return len(self.masks_dirs)



class brats_infer(data.Dataset):
	def __init__(self, mode, data_root, modality, transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms
		self.modality = modality
		# self.mapping = {
		#     0: 0,
		#     255: 1              
		# }

		#initilializing imgs with binary labels for classification



		self.images_path = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.images_dirs = glob.glob(self.images_path + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")
		self.image_name_list = os.listdir(self.images_path)


		self.images_dirs.sort()
		self.masks_dirs.sort()
		self.image_name_list.sort()

	def __getitem__(self, index):
		thresh = 0.5
		fn = lambda x : 1 if x > thresh else 0
		image = Image.open(self.images_dirs[index]).convert('L')
		mask = Image.open(self.masks_dirs[index]).convert('L')
		mask = mask.point(fn, mode='1')
		mask = torch.from_numpy(np.asarray(mask,dtype='uint8'))
		image_name = self.image_name_list[index]
		# mask_array = np.array(mask)



		# mask = torch.from_numpy(mask_array).long() # this is for my dataset(lv)

		if self.transforms is not None:
			image = self.transforms(image)

#             mask = self.transforms(mask)

#         print(np.unique(mask.numpy()))
		return (image, mask, image_name)

	def __len__(self):
		return len(self.images_dirs)


class brats_2modality(data.Dataset):
	def __init__(self, mode, data_root, modality=['T1','T2'], transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms



		self.images_path_1 = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.images_path_2 = data_root + '{}/MR_{}/'.format(mode ,modality[1])
		self.masks_path = data_root + '{}/GT/'.format(mode)
		self.images_dirs_1 = glob.glob(self.images_path_1 + "*.png")
		self.images_dirs_2 = glob.glob(self.images_path_2 + "*.png")
		self.masks_dirs = glob.glob(self.masks_path + "*.png")

			
		self.images_dirs_1.sort()
		self.images_dirs_2.sort()
		self.masks_dirs.sort()

		self.items = []
		for image_dir_1, image_dir_2, mask_dir in zip(self.images_dirs_1, self.images_dirs_2, self.masks_dirs):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0

			item = (image_dir_1, image_dir_2,label)
			self.items.append(item)


	def __getitem__(self, index):

		image_dir_1, image_dir_2, label = self.items[index]
		
		image_1 = Image.open(image_dir_1).convert('L')
		image_2 = Image.open(image_dir_2).convert('L')

		label = torch.squeeze(torch.Tensor([label])).long()


		if self.transforms is not None:
			image_1 = self.transforms(image_1)
			image_2 = self.transforms(image_2)

		image_tensor = torch.cat([image_1, image_2], dim=0)

		return (image_tensor, label)
	
	

	def __len__(self):
		return len(self.masks_dirs)

class brats_4modality(data.Dataset):
	def __init__(self, mode, data_root, modality=['T1','T2', 'T1c', 'Flair'], transforms=None):
		
		self.data_root = data_root
		self.transforms = transforms



		self.images_path_1 = data_root + '{}/MR_{}/'.format(mode ,modality[0])
		self.images_path_2 = data_root + '{}/MR_{}/'.format(mode ,modality[1])
		self.images_path_3 = data_root + '{}/MR_{}/'.format(mode ,modality[2])
		self.images_path_4 = data_root + '{}/MR_{}/'.format(mode ,modality[3])
		self.masks_path = data_root + '{}/GT/'.format(mode)

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

		self.items = []
		for image_dir_1, image_dir_2, image_dir_3, image_dir_4,mask_dir in zip(self.images_dirs_1, self.images_dirs_2, self.images_dirs_3, self.images_dirs_4, self.masks_dirs):
			mask = Image.open(mask_dir).convert('L')
			mask_array = np.array(mask)
			if np.any(mask_array):
				label = 1
			else:
				label = 0

			item = (image_dir_1, image_dir_2, image_dir_3, image_dir_4,label)
			self.items.append(item)


	def __getitem__(self, index):

		image_dir_1, image_dir_2, image_dir_3, image_dir_4,label = self.items[index]
		
		image_1 = Image.open(image_dir_1).convert('L')
		image_2 = Image.open(image_dir_2).convert('L')
		image_3 = Image.open(image_dir_3).convert('L')
		image_4 = Image.open(image_dir_4).convert('L')

		label = torch.squeeze(torch.Tensor([label])).long()


		if self.transforms is not None:
			image_1 = self.transforms(image_1)
			image_2 = self.transforms(image_2)
			image_3 = self.transforms(image_3)
			image_4 = self.transforms(image_4)

		image_tensor = torch.cat([image_1, image_2, image_3, image_4], dim=0)

		return (image_tensor, label)
	
        

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
# 			if len(self.config.modality)==1:
				validation_set_infer = brats_infer('val', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
				test_set_infer = brats_infer('test', self.config.data_root_infer,
									modality = self.config.modality,
									transforms=self.input_transform
								   )
			elif len(self.config.modality)==2:

				train_set= brats_2modality('train', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
				validation_set = brats_2modality('val', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)

			elif len(self.config.modality)==4:

				train_set= brats_4modality('train', self.config.data_root,
									modality = self.config.modality,
									transforms=self.input_transform)
				validation_set = brats_4modality('val', self.config.data_root,
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
            

		
# 		elif self.config.mode == 'test':
# # 			self.config.batch_size = 1

# 			if len(self.config.modality)==1:
# 				validation_set_infer = brats_infer('val', self.config.data_root_infer,
# 									modality = self.config.modality,
# 									transforms=self.input_transform
# 								   )
# 				test_set_infer = brats_infer('test', self.config.data_root_infer,
# 									modality = self.config.modality,
# 									transforms=self.input_transform
# 								   )
				
# 			elif len(self.config.modality)==2:

# 				validation_set_infer = brats_2modality_infer('val', self.config.data_root_infer,
# 									modality = self.config.modality,
# 									transforms=self.input_transform)
# 				test_set_infer = brats_2modality_infer('test', self.config.data_root_infer,
# 									modality = self.config.modality,
# 									transforms=self.input_transform)

# 			elif len(self.config.modality)==4:

# 				validation_set_infer = brats_4modality_infer('val', self.config.data_root_infer,
# 									modality = self.config.modality,
# 									transforms=self.input_transform)
# 				test_set_infer = brats_4modality_infer('test', self.config.data_root_inder,
# 									modality = self.config.modality,
# 									transforms=self.input_transform)

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

	train_set = brats('train', "../../brats_classification/",['T1'], transforms.Compose([transforms.ToTensor()]))

	# # # valid_set = CellSeg('val', "../../Dataset/",
	# # #                            transforms.Compose([transforms.ToTensor()]))
	# # # valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

	train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

	for image, label in train_loader:
		print(label)
	#     
