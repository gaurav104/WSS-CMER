import time
import logging
import torch.nn.functional as F
import numpy as np
import os
import glob
import torch

from PIL import Image


def timeit(f):
	""" Decorator to time Any Function """

	def timed(*args, **kwargs):
		start_time = time.time()
		result = f(*args, **kwargs)
		end_time = time.time()
		seconds = end_time - start_time
		logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
										(f.__name__, seconds, seconds / 60, seconds / 3600))
		return result

	return timed


def print_cuda_statistics():
	logger = logging.getLogger("Cuda Statistics")
	import sys
	from subprocess import call
	import torch
	logger.info('__Python VERSION:  {}'.format(sys.version))
	logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
	logger.info('__CUDA VERSION')
	call(["nvcc", "--version"])
	logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
	logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
	logger.info('__Devices')
	call(["nvidia-smi", "--format=csv",
		  "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
	logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
	logger.info('Available devices  {}'.format(torch.cuda.device_count()))
	logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))

def normalize_sample_wise(x):

	x_std, x_mean = torch.std_mean(x, dim=(-1,-2), keepdim=True)
	x_normed = (x - x_mean)/(x_std+1e-12)
	return x_normed

def one_hot_encoder(input_tensor, num_cls):
	input_tensor = F.one_hot(input_tensor, num_cls)
	input_tensor = input_tensor.permute(0,3 ,1, 2)
	input_tensor = (input_tensor>0.5).float()

	return input_tensor

def image_array(root, img_name):
	img_dir = os.path.join(root, img_name)

	img_array = np.array(Image.open(img_dir).convert('L'))
	img_array = np.expand_dims(img_array, axis = -1)

	return img_array


def reconstruct3D(root, patient_id):

	patient_slides = patient_id + "*.png"
	patient_slides_dirs = glob.glob(os.path.join(root, patient_slides))

	# print(patient_slides_dirs)

	patent_slides_dirs = sorted(patient_slides_dirs)

	slides_nparray_list = []
	for patient_slide_dir in patient_slides_dirs:
		slide_nparray = np.array(Image.open(patient_slide_dir).convert('L'))

		exp_slide_nparray = np.expand_dims(slide_nparray, axis = 0)
		# print(type())

		slides_nparray_list.append(exp_slide_nparray)

	# print(len(slides_nparray_list))
	# print(slides_nparray_list[0])
	slides_nparray_reconstructed = np.concatenate(tuple(slides_nparray_list), axis = 0)

	return slides_nparray_reconstructed

