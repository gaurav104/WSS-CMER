import numpy as np
from tqdm import tqdm
import shutil
import os
import glob
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.backends import cudnn
from torch.autograd import Variable
import subprocess
import torchvision

#importing network
from graphs.models.unet_cam import UNetCAM

#importing data loader and transformations
from datasets.brats import BratsLoader , All, Flip, Rotate, Translate, Scale, Resize

from tensorboardX import SummaryWriter

from utils.metrics import AverageMeter, cls_accuracy 
from graphs.losses.CMER_loss import MapLossL2Norm
from medpy.metric.binary import dc, assd

from utils.CAM_utils import max_norm, save_cam, CAM
from utils.misc import reconstruct3D, image_array, normalize_sample_wise
import statistics

from agents.base import BaseAgent

cudnn.benchmark = True

class WSCMER(BaseAgent):
	def __init__(self, config):
		super().__init__(config)

		self.is_cuda = torch.cuda.is_available()
		# Construct the flag and make sure that cuda is available
		self.cuda = self.is_cuda & self.config.cuda

		if self.cuda:
			torch.cuda.manual_seed_all(self.config.seed)
			self.device = torch.device("cuda")
			torch.cuda.set_device(self.config.gpu_device)
			self.logger.info("Operation will be on *****GPU-CUDA***** ")
# 			print_cuda_statistics()

		else:
			self.device = torch.device("cpu")
			torch.manual_seed(self.config.seed)
			self.logger.info("Operation will be on *****CPU***** ")

		self.num_models = self.config.num_models 
		self.models = []
		self.optimizers = []
		self.num_params = []
		self.schedulers = []

		if len(self.config.model_archs)==1:
			self.config.model_archs = self.num_models*self.config.model_archs

		# define models
		self.num_cls = self.config.num_classes
		self.in_ch = self.config.input_channels

		for i in range(self.num_models):
			if self.config.model_archs[i] == 'UNet':
				model = UNetCAM(self.in_ch[i], self.num_cls, self.config.downsize_nb_filters_factor, self.config.dropout)
				model.to(self.device)
				self.models.append(model)
			# elif self.config.model_archs[i] == 'UNet':
			# 	model = UNetCAM(self.in_ch[i], self.num_cls, self.config.downsize_nb_filters_factor, self.config.dropout)
			# 	# model.to(self.device)
			# 	self.models.append(model)
			else:
				None

			optimizer = torch.optim.AdamW(model.parameters(),
			lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
			self.optimizers.append(optimizer)

			model_parameters = filter(lambda p: p.requires_grad, model.parameters())
			num_params = sum([np.prod(p.size()) for p in model_parameters])
			self.num_params.append(num_params)

		# define data_loader
		self.data_loader = BratsLoader(self.config)

		# define loss
		self.loss_ce = nn.CrossEntropyLoss()
		self.loss_kl = nn.KLDivLoss(reduction='batchmean')
		self.loss_map = MapLossL2Norm()


		self.transformations = {'all' : All(),
					'flipping': HFlip(),
					'rotation': Rotate(),
					'translation': Translate(),
					'scaling': Resize()}


		# initialize counter
		self.current_epoch = 0
		self.current_iteration = 0
		self.best_metric_values = [0]*self.num_models

		self.loss_ce = self.loss_ce.to(self.device)
		self.loss_kl = self.loss_kl.to(self.device)

		# Model Loading from the latest checkpoint if not found start from scratch.
		if self.config.mode == 'train':
			for i in range(self.num_models):
				self.load_checkpoint(self.config.checkpoint_file, model_index=i)

		# Loading a pretrained model
		if self.config.pretrained and self.config.mode == 'train' and self.current_epoch == 0:
			for i in range(self.num_models):
				self.pretrained_model_load('{}_model_best.pth.tar', model_index=i)

		self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.exp_name)
		

	def save_checkpoint(self,is_best=0, model_index=1):
		"""
		Saving the latest checkpoint of the training
		:param filename: filename which will contain the state
		:param is_best: flag is it is the best model
		:return:
		"""
		state = {
		'epoch': self.current_epoch + 1,
		'iteration': self.current_iteration,
		'state_dict': self.models[model_index].state_dict(),
		'optimizer': self.optimizers[model_index].state_dict(),
		'best_metric_value': self.best_metric_values[model_index],
		'num_of_trainable_params': self.num_params[model_index]
		}

		filename = "{}_checkpoint.pth.tar".format(model_index)
		#self.best_metric_value is to be defined(e.g. Accuracy) which can be used for saving the best model
		# Save the state
		torch.save(state, self.config.checkpoint_dir + filename)
		# If it is the best copy it to another file 'model_best.pth.tar'
		if is_best:
			shutil.copyfile(self.config.checkpoint_dir + filename,
							self.config.checkpoint_dir + '{}_model_best.pth.tar'.format(model_index))

	def load_checkpoint(self, filename, model_index):

		filename = self.config.checkpoint_dir + filename.format(model_index) #configs checkpoint folder file name to be changed
		try:
			self.logger.info("Loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			self.best_metric_values[model_index] = checkpoint['best_metric_value']
			self.current_epoch = checkpoint['epoch']
			self.current_iteration = checkpoint['iteration']
			self.models[model_index].load_state_dict(checkpoint['state_dict'])
			self.optimizers[model_index].load_state_dict(checkpoint['optimizer'])
			self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
		except OSError as e:
			self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
			self.logger.info("**First time to train**")
 
	def pretrained_model_load(self, filename, model_index):
		project_path = "/home/gp104/projects/def-josedolz/gp104/project_wss_brats/project_wss/"
		filename = os.path.join(project_path, self.config.pretrained_model,"checkpoints",filename.format(model_index))
		# try:
		try:
			self.logger.info("Loading pretrained model '{}'".format(filename))
			checkpoint = torch.load(filename)
			self.models[model_index].load_state_dict(checkpoint['state_dict'])
			self.logger.info("pretrained_model loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
		except OSError as e:
			self.logger.info("Pretrained model not loaded '{}'. Skipping...".format(self.config.checkpoint_dir))
			self.logger.info("**Training without a pretrained model**")

	def run(self):
		assert self.config.mode in ['train', 'test', 'random']
		try:
			if self.config.mode == 'test':
				self.test()
			else:
				self.train()
		except KeyboardInterrupt:
			self.logger.info("You have entered CTRL+C.. Wait to finalize")

	def train(self):
		"""
		Main training function, with per-epoch model saving
		"""
		for epoch in range(self.current_epoch, self.config.max_epoch):
			self.current_epoch = epoch
			self.train_one_epoch()
		self.current_epoch = self.config.max_epoch

		return

	def train_one_epoch(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		transform = self.transformations[self.config.transformation]
		ce = [] 
		accuracy = []
		for i in range(self.num_models):
			self.models[i].train()
			ce.append(AverageMeter())
			accuracy.append(AverageMeter())
		
		for x, y, _ in tqdm_batch:

			if self.cuda:
				x, y = x.pin_memory().cuda(), y.pin_memory().cuda(non_blocking=True)
			x, y = Variable(x), Variable(y)
			N,C,H,W = x.size()

			x1 = x
			preds1 = []
			cams1 = []
			norm_x1 = normalize_sample_wise(x1)
			
			for i in range(self.num_models):
				pred1, cam1 = self.models[i](norm_x1[:,i:i+1,:,:])
				pred1 = torch.squeeze(pred1, -1)
				pred1 = torch.squeeze(pred1, -1)
				preds1.append(pred1)
				cams1.append(max_norm(cam1))

			x2, cams1 = transform(x1, cams1)
			norm_x2 = normalize_sample_wise(x2)

			preds2 = []
			cams2 = []

			for i in range(self.num_models):
				pred2, cam2 = self.models[i](norm_x2[:,i:i+1,:,:])
				pred2 = torch.squeeze(pred2, -1)
				pred2= torch.squeeze(pred2, -1)
				preds2.append(pred2)
				cams2.append(max_norm(cam2))

			ce_losses = []
			er_losses = []
			kl_losses = []
			cmer_losses = []
			accuracies = []

			for i in range(self.num_models):
				ce_loss_1 = 0
				ce_loss_2 = 0
				kl_loss_1 = 0
				kl_loss_2 = 0
				cmer_loss_1 = 0
				cmer_loss_2 = 0
				cur_loss = 0

				ce_loss_1 = self.loss_ce(preds1[i], y)
				ce_loss_2 = self.loss_ce(preds2[i], y)

				er_loss = torch.mean(torch.pow(cams1[i][:,1:,:,:] - cams2[i][:,1:,:,:],2)*y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

				for j in range(self.num_models):
					if i!=j:
						kl_loss_1 += (self.config.temp**2)*self.loss_kl(F.log_softmax(preds1[i]/self.config.temp,dim=1),
													F.softmax(Variable(preds1[j]/self.config.temp), dim=1))
						kl_loss_2 += (self.config.temp**2)*self.loss_kl(F.log_softmax(preds2[i]/self.config.temp,dim=1),
													F.softmax(Variable(preds2[j]/self.config.temp), dim=1))

						label_mask_1 = (torch.argmax(preds2[j].detach(), dim=-1)==1).float() * y
						p1 = cams1[i]
						q1 = cams2[j].detach()

						label_mask_2 = (torch.argmax(preds1[j].detach(), dim=-1)==1).float() * y
						p2 = cams2[i]
						q2 = cams1[j].detach()

						cmer_loss_1 += self.loss_map(p1[:,1:,:,:], q1[:,1:,:,:], label_mask_1)
						cmer_loss_2 += self.loss_map(p2[:,1:,:,:], q2[:,1:,:,:], label_mask_2)

				kl_loss = 0.5*(kl_loss_1 + kl_loss_2)
				kl_loss = kl_loss / (self.num_models-1)
				cmer_loss = 0.5 * (cmer_loss_1 + cmer_loss_2)
				cmer_loss = cmer_loss/(self.num_models-1)
				
				ce_loss = 0.5*(ce_loss_1 + ce_loss_2)
				cur_loss = ce_loss + (self.config.lmda *kl_loss) + er_loss + cmer_loss

				self.optimizers[i].zero_grad()
				cur_loss.backward()
				self.optimizers[i].step()

				ce_losses.append(ce_loss.item())
				er_losses.append(er_loss.item())
				kl_losses.append(kl_loss.item())
				cmer_losses.append(cmer_loss.item())
				
				iter_accuracy = cls_accuracy(preds1[i], y)
				iter_accuracy = iter_accuracy[0].cpu().numpy()
				accuracies.append(iter_accuracy)

			self.current_iteration += 1

			if self.current_iteration == 1:
				#Adding Scalars
				validation_values = self.validate()
				for i in range(self.num_models):
					self.summary_writer.add_scalars("{}_ce_loss".format(i),{"train_ce_loss":ce_losses[i], "valid_ce_loss":validation_values['ce_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_kl_loss".format(i),{"train_kl_loss":kl_losses[i], "valid_kl_loss":validation_values['kl_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_er_loss".format(i),{"train_er_loss":er_losses[i]}, self.current_iteration)
					self.summary_writer.add_scalars("{}_accuracy".format(i),{"train_accuracy":accuracies[i], "valid_accuracy":validation_values['accuracy'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_cmer_loss".format(i),{"train_cmer_loss":cmer_losses[i]}, self.current_iteration)


			if self.current_iteration%200 == 0:
				#Adding Scalars
				validation_values = self.validate()
				for i in range(self.num_models):
					print("{} Epoch-".format(i) + str(self.current_epoch))
					print("{} Train CE loss: ".format(i) + str(ce_losses[i]))
					print("{} Validation CE loss: ".format(i) + str(validation_values['ce_loss'][i]))
					print("{} Train Accuracy: ".format(i) + str(accuracies[i]))
					print("{} Validation Accuracy: ".format(i) + str(validation_values['accuracy'][i])+ "\n")
					self.summary_writer.add_scalars("{}_ce_loss".format(i),{"train_ce_loss":ce_losses[i], "valid_ce_loss":validation_values['ce_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_kl_loss".format(i),{"train_kl_loss":kl_losses[i], "valid_kl_loss":validation_values['kl_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_er_loss".format(i),{"train_er_loss":er_losses[i]}, self.current_iteration)
					self.summary_writer.add_scalars("{}_accuracy".format(i),{"valid_accuracy":validation_values['accuracy'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_cmer_loss".format(i),{"train_cmer_loss":cmer_losses[i]}, self.current_iteration)

					is_best = validation_values['accuracy'][i] > self.best_metric_values[i]

					if(is_best):
						self.best_metric_values[i] = validation_values['accuracy'][i]
					
					self.save_checkpoint(is_best=is_best, model_index = i)
		tqdm_batch.close()
		return 

	def validate(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.validation_loader, total=self.data_loader.validation_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		ce = []
		kl = []
		accuracy = []
		for i in range(self.num_models):
			# Set the model to be in eval mode (for batchnorm and dropout)
			self.models[i].eval()

			ce.append(AverageMeter())
			kl.append(AverageMeter())
			accuracy.append(AverageMeter())
		
		for x, y, _ in tqdm_batch:
			if self.cuda:
				x, y = x.pin_memory().cuda(), y.pin_memory().cuda(non_blocking=True)
			x, y = Variable(x), Variable(y)
			
			preds = []
			cams = []
			norm_x = normalize_sample_wise(x)
			for i in range(self.num_models):
				pred, cam = self.models[i](norm_x[:,i:i+1,:,:])
				pred = torch.squeeze(pred,-1)
				pred = torch.squeeze(pred, -1)
				preds.append(pred)
				cams.append(cam)

			for i in range(self.num_models):
				ce_loss = self.loss_ce(preds[i], y)
				kl_loss = 0
				for j in range(self.num_models):
					if i!=j:
						kl_loss += (self.config.temp**2)*self.loss_kl(F.log_softmax(preds[i]/self.config.temp,dim=1),
													F.softmax(Variable(preds[j]/self.config.temp), dim=1))
				kl_loss = kl_loss/(self.num_models - 1)

				cur_loss = ce_loss + self.config.lmda * kl_loss

				iter_accuracy = cls_accuracy(preds[i], y)
				iter_accuracy = iter_accuracy[0].cpu().numpy()

				ce[i].update(ce_loss.item())
				kl[i].update(kl_loss.item())
				accuracy[i].update(iter_accuracy)

		tqdm_batch.close()

		return {'accuracy': [acc.val for acc in accuracy],
			'ce_loss': [loss.val for loss in ce],
			'kl_loss': [loss.val for loss in kl]}

	def test(self):
		for model_index in range(self.config.num_models):
			self.infer('val', model_index)
			self.infer('test', model_index)

	def save_predictions(self, data):
		if self.current_epoch == self.config.max_epoch:
			for i in range(self.num_models):
				self.load_checkpoint("{}_model_best.pth.tar", model_index=i)
				self.current_epoch = self.config.max_epoch
				project_path = "../experiments/"
				dest_tar = os.path.join(project_path, self.config.exp_name, '{}_predictions_{}.tar'.format(i,data))
				if os.path.isfile(dest_tar):
					print("Predictions already saved!.")
					continue

				if data =='val':
					tqdm_batch = tqdm(self.data_loader.validation_infer_loader, total=self.data_loader.validation_infer_iterations,
					desc="Epoch-{}-".format(1))
				elif data =='test':
					tqdm_batch = tqdm(self.data_loader.test_infer_loader, total=self.data_loader.test_infer_iterations,
					desc="Epoch-{}-".format(1))
				else:
					return

				gt_dir = os.path.join(self.config.data_root_infer, "{}/GT/".format(data))
				dest_root_cam = os.path.join(self.config.data_root_infer, "{}/{}/{}_GT_CAM_{}/".format(self.config.exp_name, data, i,self.config.modality[i]))
				dest_root_gradcam = os.path.join(self.config.data_root_infer, "{}/{}/{}_GT_GRADCAM_{}/".format(self.config.exp_name,data, i,self.config.modality[i]))
				dest_root_gradcampp = os.path.join(self.config.data_root_infer, "{}/{}/{}_GT_GRADCAMPP_{}/".format(self.config.exp_name, data, i,self.config.modality[i]))

				if not os.path.exists(dest_root_cam):
					os.makedirs(dest_root_cam)
				if not os.path.exists(dest_root_gradcam):
					os.makedirs(dest_root_gradcam)
				if not os.path.exists(dest_root_gradcampp):
					os.makedirs(dest_root_gradcampp)

				self.models[i].eval()
				cam = CAM(self.models[i], self.models[i].backbone)
				for x, y, img_name in tqdm_batch:
					if self.cuda:
						x, y = x.pin_memory().cuda(), y.pin_memory().cuda(non_blocking=True)
					x, y = Variable(x), Variable(y)

					x = normalize_sample_wise(x)

					cam_map, grad_cam_map, grad_cam_pp_map, logit = cam(x[:,i:i+1,:,:], class_idx=1)

					prob = F.softmax(logit,dim=1)

					idx = torch.argmax(prob,1)

					idx = torch.squeeze(idx, 0).item() 
					cam_map, grad_cam_map, grad_cam_pp_map = torch.squeeze(cam_map), torch.squeeze(grad_cam_map), torch.squeeze(grad_cam_pp_map)

					save_cam(cam = cam_map, idx = idx, dest_root=dest_root_cam, img_name=img_name[0])
					save_cam(cam = grad_cam_map, idx = idx, dest_root=dest_root_gradcam, img_name=img_name[0])
					save_cam(cam = grad_cam_pp_map, idx = idx, dest_root=dest_root_gradcampp, img_name=img_name[0])

				dest_tar = os.path.join(project_path, self.config.exp_name, '{}_predictions_{}.tar'.format(i, data))
				dir_to_tar = os.path.join("brats_data", self.config.exp_name, data)
				list_files = subprocess.run(["tar", "-C", data_path, "-cf", dest_tar, dir_to_tar], check=True)
				print("Predictions compression and saving executed with : %d" % list_files.returncode)

		else:
			print("Skipping predictions saving")
			return

	def infer(self, data, model_index):
		
		"""
			Inference function
		"""
		pass


	def finalize(self):
		"""
		Finalize all the operations 

		"""
		print("Please wait while finalizing the operation.. Thank you")
		for i in range(self.num_models):
			self.save_checkpoint(model_index=i)

		self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir,"all_scalars.json"))
		self.summary_writer.flush()
		self.summary_writer.close()


