import numpy as np
from tqdm import tqdm
import shutil
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision

#importing network
from graphs.models.unet import UNet

#importing data loader
from datasets.brats_seg import BratsLoader, SegAugPipeline # importing data Loader

#importing loss functions and optimizers
# from graphs.losses.bce import BinaryCrossEntropy
from kornia.losses import DiceLoss
# from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from utils.train_utils import export_jsondump
from utils.metrics import AverageMeter, AverageMeterList, Dice
# from utils.misc import print_cuda_statistics

import time
# from utils.train_utils 
import statistics

from agents.base import BaseAgent

cudnn.benchmark = True

class SegAgent(BaseAgent):

	def __init__(self, config):
		super().__init__(config)

		# define models
		self.num_cls = self.config.num_classes
		self.in_ch = self.config.input_channels
		self.model = UNet(self.in_ch, self.num_cls, 2)
		model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.num_params = sum([np.prod(p.size()) for p in model_parameters])

		# define data_loader
		self.data_loader = BratsLoader(self.config)

		# define loss
#         weights = [0.1, 0.9]
#         class_weights = torch.FloatTensor(weights)
		self.loss_ce = nn.CrossEntropyLoss()
		self.loss_dice = DiceLoss()
		# self.loss_inv_dice = InvSoftDiceLoss()

		# define optimizers for both generator and discriminator
		self.optimizer = torch.optim.AdamW(self.model.parameters(),
			lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

		# initialize counter
		self.current_epoch = 0
		self.current_iteration = 0
		self.best_metric_value = float('inf') # lowest value

		# Check is cuda is available or not
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


		self.model = self.model.to(self.device)
		self.loss_ce = self.loss_ce.to(self.device)
		self.loss_dice = self.loss_dice.to(self.device)
		# self.loss_inv_dice = self.loss_inv_dice.to(self.device)

		# Model Loading from the latest checkpoint if not found start from scratch.

		self.load_checkpoint(self.config.checkpoint_file)
		self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=self.config.exp_name)

		

	def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
		"""
		Saving the latest checkpoint of the training
		:param filename: filename which will contain the state
		:param is_best: flag is it is the best model
		:return:
		"""
		state = {
			'epoch': self.current_epoch + 1,
			'iteration': self.current_iteration,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'best_metric_value': self.best_metric_value,
			'num_of_trainable_params': self.num_params
		}

		#self.best_metric_value is to be defined(e.g. Accuracy) which can be used for saving the best model
		# Save the state
		torch.save(state, self.config.checkpoint_dir + filename)
		# If it is the best copy it to another file 'model_best.pth.tar'
		if is_best:
			shutil.copyfile(self.config.checkpoint_dir + filename,
							self.config.checkpoint_dir + 'model_best.pth.tar')




	def load_checkpoint(self, filename):

		filename = self.config.checkpoint_dir + filename

		try:
			self.logger.info("Loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			self.best_metric_value = checkpoint['best_metric_value']
			self.current_epoch = checkpoint['epoch']
			self.current_iteration = checkpoint['iteration']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
		except OSError as e:
			self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
			self.logger.info("**First time to train**")


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
			# self.scheduler.step(epoch)
			start = time.time()
			self.train_one_epoch()
			end = time.time()
			time_taken = end-start
			print('Average time per epoch(minutes): ', time_taken%60)
			self.save_checkpoint()
			# valid_mIoU, valid_mDice  = self.validate()
			# self.scheduler.step(valid_loss)

			



	def train_one_epoch(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		augment = SegAugPipeline()

		# Set the model to be in training mode (for batchnorm)
		self.model.train()
		

		# Initialize average meters of losses
		# ce = AverageMeter()
		dice = AverageMeter()
		ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		# accuracy = AverageMeter()
		dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:
			

			if self.cuda:
				x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
			x , y= augment(x,y)
			x, y = Variable(x), Variable(y)


			# model
			pred = self.model(x)
# 			pred = torch.squeeze(pred)
			
			# loss
			ce_loss = self.loss_ce(pred, y)
			dice_loss = self.loss_dice(pred, y)
			# inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
			cur_loss =  ce_loss+dice_loss
#             if np.isnan(float(cur_loss.item())):
#                 raise ValueError('Loss is nan during training...')

			# optimizer
			self.optimizer.zero_grad()
			cur_loss.backward()
			self.optimizer.step()
			
			ce.update(ce_loss.item())
			# dice.update(dice_loss.item())
			# inv_dice.update(inv_dice_loss.item())


			##calculate accuracy and update its average list
			# iter_accuracy = cls_accuracy(pred, y)
			iter_dice_coeff =  Dice(pred, y, self.num_cls)
			iter_dice_coeff = iter_dice_coeff.cpu().numpy()
			# iter_iou = IoU(pred,y, self.num_cls)
			# iter_accuracy = iter_accuracy[0].cpu().numpy()
			# accuracy.update(iter_accuracy)

			dice_coeff_hard.update(iter_dice_coeff)
			# iou.update(iter_iou.cpu().numpy())

			# pred_max = torch.sigmoid(pred) > 0.5
			# metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

			self.current_iteration += 1

			if self.current_iteration == 1:
				validation_values = self.validate()
				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("dice_loss",{"train_dice_loss":dice_loss.item(), "valid_dice_loss":validation_values['dice_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("total_loss",{"train_total_loss":cur_loss.item(), "valid_total_loss":validation_values['cur_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("dice",{"train_dice":iter_dice_coeff[1] , "valid_dice":validation_values['dice'][1]}
					, self.current_iteration)




			if self.current_iteration%400 == 0:
				validation_values = self.validate()
				print("Epoch-" + str(self.current_epoch))
				print("Train CE loss: " + str(ce_loss.item()))
				print("Validation CE loss: " + str(validation_values['ce_loss']))
				print("Train Dice: " + str(iter_dice_coeff[1]))
				print("Validation Dice: " + str(validation_values['dice'][1])+ "\n")

				#Adding Scalars
				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("dice_loss",{"train_dice_loss":dice_loss.item(), "valid_dice_loss":validation_values['dice_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("total_loss",{"train_total_loss":cur_loss.item(), "valid_total_loss":validation_values['cur_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("dice",{"train_dice":iter_dice_coeff[1], "valid_dice":validation_values['dice'][1]}
					, self.current_iteration)

				is_best = validation_values['cur_loss'] < self.best_metric_value

				if(is_best):
					self.best_metric_value = validation_values['cur_loss']
					self.save_checkpoint(is_best=is_best)
# 				self.summary_writer.add_hparams(self.config,{'accuracy':self.best_metric_value})
				
				# self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)



		tqdm_batch.close()
		print("Epoch Average Dice- {}".format(dice_coeff_hard.val[1]))
		return 

	def validate(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.validation_loader, total=self.data_loader.validation_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		# Set the model to be in training mode (for batchnorm)
		self.model.eval()
		

		# Initialize average meters of losses
		# ce = AverageMeter()
		dice = AverageMeter()
		ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		# accuracy = AverageMeter()
		dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:
			if self.cuda:
				x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
			x, y = Variable(x), Variable(y)


			# model
			with torch.no_grad():
				pred = self.model(x)

# 			pred = torch.squeeze(pred)
			
			# loss
			ce_loss = self.loss_ce(pred, y)
			dice_loss = self.loss_dice(pred, y)
			# inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
			cur_loss =  ce_loss + dice_loss
#             if np.isnan(float(cur_loss.item())):
#                 raise ValueError('Loss is nan during training...')

			
			ce.update(ce_loss.item())
			dice.update(dice_loss.item())
			# inv_dice.update(inv_dice_loss.item())


			##calculate accuracy and update its average list
# 			iter_accuracy = cls_accuracy(pred, y)
			iter_dice_coeff =  Dice(pred, y, self.num_cls)
			iter_dice_coeff = iter_dice_coeff.cpu().numpy()
			# iter_iou = IoU(pred,y, self.num_cls)
			# iter_accuracy = iter_accuracy[0].cpu().numpy()
			# accuracy.update(iter_accuracy)

			dice_coeff_hard.update(iter_dice_coeff)
			# iou.update(iter_iou.cpu().numpy())

			# pred_max = torch.sigmoid(pred) > 0.5
			# metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())



			# 	#Adding Scalars

			# 	# self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)


		tqdm_batch.close()
		# print("Epoch Average Accuracy- {}".format(accuracy.val))
		return {'dice': dice_coeff_hard.val,
				'ce_loss': ce.val,
				'dice_loss':dice.val,
				'cur_loss':ce.val+dice.val}

	def finalize(self):
		"""
		Finalize all the operations of the 2 Main classes of the process the operator and the data loader
		:return:
		"""
		print("Please wait while finalizing the operation.. Thank you")
		self.save_checkpoint()

		self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir,"all_scalars.json"))
		# export_jsondump(self.summary_writer)
		self.summary_writer.flush()
		self.summary_writer.close()
		# self.data_loader.finalize()

