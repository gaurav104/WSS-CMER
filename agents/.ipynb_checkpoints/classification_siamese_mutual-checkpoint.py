import numpy as np
from tqdm import tqdm
import shutil
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision

#importing network
from graphs.models.resnet50_conv import ResNet50

#importing data loader
from datasets.brats import BratsLoader#, #ClfAugPipeline # importing data Loader

#importing loss functions and optimizers
# from graphs.losses.bce import BinaryCrossEntropy
# from kornia.losses import DiceLoss
# from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from utils.train_utils import export_jsondump
from utils.metrics import AverageMeter, cls_accuracy
# from utils.misc import print_cuda_statistics
from utils.CAM_utils import max_norm
# from utils.train_utils 
import statistics

from agents.base import BaseAgent

cudnn.benchmark = True

class ClassificationSiaMutualAgent(BaseAgent):

	def __init__(self, config):
		super().__init__(config)

		# define models
		# Check is cuda is available or not
		self.is_cuda = torch.cuda.is_available()
		# Construct the flag and make sure that cuda is available
		self.cuda = self.is_cuda & self.config.cuda

		if self.cuda:
			torch.cuda.manual_seed_all(self.config.seed)
			self.device = torch.device("cuda")
			torch.cuda.set_device(self.config.gpu_device)
			self.logger.info("Operation will be on *****GPU-CUDA***** ")
#           print_cuda_statistics()

		else:
			self.device = torch.device("cpu")
			torch.manual_seed(self.config.seed)
			self.logger.info("Operation will be on *****CPU***** ")

		self.num_models = config.num_models #to be added in the config
		self.models = []
		self.optimizers = []
		self.num_params = []
		self.schedulers = []

		self.num_cls = self.config.num_classes
		self.in_ch = self.config.input_channels #list of input channels for each model

		for i in range(self.num_models):
			model = ResNet50(self.in_ch[i], self.num_cls)
			model.to(self.device)
			self.models.append(model)

			optimizer = torch.optim.AdamW(model.parameters(),
			lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
			self.optimizers.append(optimizer)

			model_parameters = filter(lambda p: p.requires_grad, model.parameters())
			num_params = sum([np.prod(p.size()) for p in model_parameters])
			self.num_params.append(num_params)


		# self.model = ResNet50(self.in_ch, self.num_cls)
		

		# define data_loader
		self.data_loader = BratsLoader(self.config)

		# define loss
		self.loss_ce = nn.CrossEntropyLoss()
		self.loss_kl = nn.KLDivLoss(reduction='batchmean')
		# self.loss_dice = DiceLoss()
		# self.loss_inv_dice = InvSoftDiceLoss()

		# define optimizers for both generator and discriminator
		# self.optimizer = torch.optim.AdamW(self.model.parameters(),
		#     lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

		# initialize counter
		self.current_epoch = 0
		self.current_iteration = 0
		self.best_metric_values = [float('inf')]*self.num_models # high value if loss and low value for metric

		# Check is cuda is available or not
#         self.is_cuda = torch.cuda.is_available()
#         # Construct the flag and make sure that cuda is available
#         self.cuda = self.is_cuda & self.config.cuda

#         if self.cuda:
#             torch.cuda.manual_seed_all(self.config.seed)
#             self.device = torch.device("cuda")
#             torch.cuda.set_device(self.config.gpu_device)
#             self.logger.info("Operation will be on *****GPU-CUDA***** ")
# #           print_cuda_statistics()

#         else:
#             self.device = torch.device("cpu")
#             torch.manual_seed(self.config.seed)
#             self.logger.info("Operation will be on *****CPU***** ")


#         self.model = self.model.to(self.device)
		
		self.loss_ce = self.loss_ce.to(self.device)
		self.loss_kl = self.loss_kl.to(self.device)
		# self.loss_dice = self.loss_dice.to(self.device)
		# self.loss_inv_dice = self.loss_inv_dice.to(self.device)

		# Model Loading from the latest checkpoint if not found start from scratch.
		for i in range(self.num_models):
			self.load_checkpoint(self.config.checkpoint_file, model_index=i)

		# self.load_checkpoint(self.config.checkpoint_file)
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

		filename = self.config.checkpoint_dir + "{}_checkpoint.pth.tar".format(model_index) #configs checkpoint folder file name to be changed

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
			self.train_one_epoch()

			# valid_mIoU, valid_mDice  = self.validate()
			# self.scheduler.step(valid_loss)

			



	def train_one_epoch(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		augment = ClfAugPipeline()

		ce = []

		accuracy = []

		# Set the model to be in training mode (for batchnorm)
		for i in range(self.num_models):
			self.models[i].train()
			ce.append(AverageMeter())
			accuracy.append(AverageMeter())
		# self.model.train()
		

		# Initialize average meters of losses
		# ce = AverageMeter()
		# dice = AverageMeter()
		# ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		# accuracy = AverageMeter()
		# dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:

			

			if self.cuda:
				x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)

			N,C,H,W = x.size()

			x = augment(x)

			x1, y = Variable(x), Variable(y)
			scale_factor = 0.3
			x2 = F.interpolate(x1,scale_factor=scale_factor,mode='bilinear',align_corners=True)
			# model
			preds_1 = []
			cams_1 = []

			preds_2 = []
			cams_2 = []

			for i in range(self.num_models):
				pred_1, cam_1 = self.models[i](x1[:,i:i+1,:,:])
				pred_1 = torch.squeeze(pred_1)
				cam_1 = F.interpolate(max_norm(cam_1), scale_factor=scale_factor,mode='bilinear',align_corners=True)

				pred_2, cam_2 = self.models[i](x2[:,i:i+1,:,:])
				pred_2 = torch.squeeze(pred_2)
				cam_2 = max_norm(cam_2)

				preds_1.append(pred_1)
				preds_2.append(pred_2)

				cams_1.append(cam_1)
				cams_2.append(cam_2)

				# logits.append(logit.detach())

			ce_losses = []
			ec_losses = []
			kl_losses = []

			for i in range(self.num_models):
				ce_loss_1 = self.loss_ce(preds_1[i], y)
				ce_loss_2 = self.loss_ce(preds_2[i], y)
				ec_loss = torch.mean(torch.abs(cams_1[i][:,1:,:,:] - cams_2[i][:,1:,:,:])*y.view(N,1,1,1))

				ce_loss = 0.5*(ce_loss_1 + ce_loss_2) 

				kl_loss_1 = 0
				kl_loss_2 = 0
				for j in range(self.num_models):
					if i!=j:
						kl_loss_1 += (self.config.temp**2)*self.loss_kl(F.log_softmax(preds_1[i]/self.config.temp,dim=1),
													F.softmax(Variable(preds_1[j]/self.config.temp), dim=1))
												
						kl_loss_2 += (self.config.temp**2)*self.loss_kl(F.log_softmax(preds_2[i]/self.config.temp,dim=1),
													F.softmax(Variable(preds_2[j]/self.config.temp), dim=1))
				kl_loss = 0.5 * (kl_loss_1 / (self.num_models-1)) + 0.5*(kl_loss_2 / (self.num_models-1))

				curr_loss = ce_loss + (self.config.lmda*kl_loss) + ec_loss

				self.optimizers[i].zero_grad()
				curr_loss.backward()
				self.optimizers[i].step()


				# ce[i].update(ce_loss.item())
				ce_losses.append(ce_loss.item())
				ec_losses.append(ec_loss.item())
				kl_losses.append(kl_loss.item())

			self.current_iteration += 1

			if self.current_iteration == 1:
				validation_values = self.validate()
				for i in range(self.num_models):

					self.summary_writer.add_scalars("{}_ce_loss".format(i),{"train_ce_loss":ce_losses[i], "valid_ce_loss":validation_values['ce_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_kl_loss".format(i),{"train_kl_loss":kl_losses[i], "valid_kl_loss":validation_values['kl_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_ec_loss".format(i),{"train_ec_loss":ec_losses[i]}, self.current_iteration)
					self.summary_writer.add_scalars("{}_accuracy".format(i),{"valid_accuracy":validation_values['accuracy'][i]}
						, self.current_iteration)

			if self.current_iteration%200 == 0:
				validation_values = self.validate()
				for i in range(self.num_models):
					print("{} Epoch-".format(i) + str(self.current_epoch))
					print("{} Train CE loss: ".format(i) + str(ce_losses[i]))
					print("{} Validation CE loss: ".format(i) + str(validation_values['ce_loss'][i]))
	#               print("Train Accuracy: " + str(iter_accuracy))
					print("{} Validation Accuracy: ".format(i) + str(validation_values['accuracy'][i])+ "\n")

				#Adding Scalars
					self.summary_writer.add_scalars("{}_ce_loss".format(i),{"train_ce_loss":ce_losses[i], "valid_ce_loss":validation_values['ce_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_kl_loss".format(i),{"train_kl_loss":kl_losses[i], "valid_kl_loss":validation_values['kl_loss'][i]}
						, self.current_iteration)
					self.summary_writer.add_scalars("{}_ec_loss".format(i),{"train_ec_loss":ec_losses[i]}, self.current_iteration)
					self.summary_writer.add_scalars("{}_accuracy".format(i),{"valid_accuracy":validation_values['accuracy'][i]}
						, self.current_iteration)

					is_best = validation_values['ce_loss'][i] < self.best_metric_values[i]

					if(is_best):
						self.best_metric_value = validation_values['ce_loss'][i]
					
					self.save_checkpoint(is_best=is_best, model_index = i)



# 			self.current_iteration += 1

# 			if self.current_iteration == 1:
# 				validation_values = self.validate()
# 				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
# 					, self.current_iteration)
# 				self.summary_writer.add_scalars("ec_loss",{"train_ec_loss":ec_loss.item()}, self.current_iteration)
# 				self.summary_writer.add_scalars("accuracy",{"valid_accuracy":validation_values['accuracy']}
# 					, self.current_iteration)




# 			if self.current_iteration%200 == 0:
# 				validation_values = self.validate()
# 				print("Epoch-" + str(self.current_epoch))
# 				print("Train CE loss: " + str(ce_loss.item()))
# 				print("Validation CE loss: " + str(validation_values['ce_loss']))
# #               print("Train Accuracy: " + str(iter_accuracy))
# 				print("Validation Accuracy: " + str(validation_values['accuracy'])+ "\n")

# 				#Adding Scalars
# 				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
# 					, self.current_iteration)
# 				self.summary_writer.add_scalars("accuracy",{"valid_accuracy":validation_values['accuracy']}
# 					, self.current_iteration)

# 				is_best = validation_values['ce_loss'] < self.best_metric_value

# 				if(is_best):
# 					self.best_metric_value = validation_values['ce_loss']
# 				self.save_checkpoint(is_best=is_best)
# #               self.summary_writer.add_hparams(self.config,{'accuracy':self.best_metric_value})
				
# 				# self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)



		tqdm_batch.close()
# 		print("Epoch Average Accuracy- {}".format(accuracy.val))
		return 

	def validate(self):
		"""
		One epoch training function
		"""
		# Initialize tqdm
		tqdm_batch = tqdm(self.data_loader.validation_loader, total=self.data_loader.validation_iterations,
			desc="Epoch-{}-".format(self.current_epoch))

		# Set the model to be in training mode (for batchnorm)
		# self.models[model_index].eval()
		# model = self.models[model_index].eval()
		

		# Initialize average meters of losses
		# ce = AverageMeter()
		# dice = AverageMeter()
		ce = []
		kl = []
		accuracy = []

		# Set the model to be in training mode (for batchnorm)
		for i in range(self.num_models):
			self.models[i].eval()
			ce.append(AverageMeter())
			kl.append(AverageMeter())
			accuracy.append(AverageMeter())

		# ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		# accuracy = AverageMeter()
		# dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:
			if self.cuda:
				x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
			x, y = Variable(x), Variable(y)


			# model
			preds = []
			cams = []

			# ce_losses = []

			for i in range(self.num_models):
				pred, cam = self.models[i](x[:,i:i+1,:,:])
				pred = torch.squeeze(pred)

				preds.append(pred)
				cams.append(cam)

			# ce_losses = []
			# kl_losses = []

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
				# iter_dice_coeff_dice =  Dice(pred, y, self.num_cls)
				# iter_iou = IoU(pred,y, self.num_cls)
				iter_accuracy = iter_accuracy[0].cpu().numpy()

				ce[i].update(ce_loss.item())
				kl[i].update(kl_loss.item())
				accuracy[i].update(iter_accuracy)





# 			pred, cam = model(x)
# 			pred = torch.squeeze(pred)
			
# 			# loss
# 			ce_loss = self.loss_ce(pred, y)
# 			# dice_loss = self.loss_dice(pred, y)
# 			# inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
# 			cur_loss =  ce_loss 
# #             if np.isnan(float(cur_loss.item())):
# #                 raise ValueError('Loss is nan during training...')

			
# 			ce.update(ce_loss.item())
# 			# dice.update(dice_loss.item())
# 			# inv_dice.update(inv_dice_loss.item())


# 			##calculate accuracy and update its average list
# 			iter_accuracy = cls_accuracy(pred, y)
# 			# iter_dice_coeff_dice =  Dice(pred, y, self.num_cls)
# 			# iter_iou = IoU(pred,y, self.num_cls)
# 			iter_accuracy = iter_accuracy[0].cpu().numpy()
# 			accuracy.update(iter_accuracy)

			# dice_coeff_hard.update(iter_dice_coeff_dice.cpu().numpy())
			# iou.update(iter_iou.cpu().numpy())

			# pred_max = torch.sigmoid(pred) > 0.5
			# metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())



			#   #Adding Scalars

			#   # self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)


		tqdm_batch.close()
		# print("Epoch Average Accuracy- {}".format(accuracy.val))
		return {'accuracy': [acc.val for acc in accuracy],
				'ce_loss': [loss.val for loss in ce],
				'kl_loss': [loss.val for loss in kl]}

	def finalize(self):
		"""
		Finalize all the operations of the 2 Main classes of the process the operator and the data loader
		:return:
		"""
		print("Please wait while finalizing the operation.. Thank you")
		for i in range(self.num_models):
			self.save_checkpoint(model_index=i)
# 		self.save_checkpoint()

		self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir,"all_scalars.json"))
		# export_jsondump(self.summary_writer)
		self.summary_writer.flush()
		self.summary_writer.close()
		# self.data_loader.finalize()

