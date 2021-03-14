import numpy as np
from tqdm import tqdm
import shutil
import os
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision

#importing network
from graphs.models.resnet50_conv import ResNet50

#importing data loader
from datasets.brats import BratsLoader, All, HFlip, Rotate, Translate, Scale, ClfAugPipeline # importing data Loader

#importing loss functions and optimizers
# from graphs.losses.bce import BinaryCrossEntropy
# from kornia.losses import DiceLoss
# from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from utils.train_utils import export_jsondump
from utils.metrics import AverageMeter, cls_accuracy, Dice
from utils.surface_metrics import compute_surface_distances, compute_average_surface_distance_2
# from utils.misc import print_cuda_statistics

from utils.CAM_utils import max_norm, save_cam
from utils.misc import reconstruct3D
# from utils.train_utils 
import statistics

from agents.base import BaseAgent

cudnn.benchmark = True

class ClassificationSiaAgent(BaseAgent):

	def __init__(self, config):
		super().__init__(config)

		# define models
		self.num_cls = self.config.num_classes
		self.in_ch = self.config.input_channels

		if self.config.model_arch == 'ResNet50':
			self.model = ResNet50(self.in_ch, self.num_cls)
		else:
			None

		model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.num_params = sum([np.prod(p.size()) for p in model_parameters])

		# define data_loader
		self.data_loader = BratsLoader(self.config)

		# define loss
		self.loss_ce = nn.CrossEntropyLoss()
		# self.loss_dice = DiceLoss()
		# self.loss_inv_dice = InvSoftDiceLoss()
		self.transformations = {'all' : All(),
					'flipping': HFlip(),
					'rotation': Rotate(),
					'translation': Translate(),
					'scaling': Scale()}

		# define optimizers for both generator and discriminator
		self.optimizer = torch.optim.AdamW(self.model.parameters(),
			lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

		# initialize counter
		self.current_epoch = 0
		self.current_iteration = 0
		self.best_metric_value = 0 # lowest value

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
		# self.loss_dice = self.loss_dice.to(self.device)
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
				# self.save_predictions('val')
				# self.save_predictions('test')

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


		transform = self.transformations[self.config.transformation]
		augment = ClfAugPipeline()

		# Set the model to be in training mode (for batchnorm)
		self.model.train()
		

		# Initialize average meters of losses
		# ce = AverageMeter()
		# dice = AverageMeter()
		ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		accuracy = AverageMeter()
		# dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:
			

			if self.cuda:
				x, y = x.pin_memory().cuda(), y.pin_memory().cuda()
			
			N,C,H,W = x.size()
			x = augment(x)
			

			x1, y = Variable(x), Variable(y.contiguous())


			# model
			pred1, cam1 = self.model(x1)
# 			print(pred.size())

			x2, cam1 = transform(x1, max_norm(cam1))


			pred2, cam2 = self.model(x2)
			cam2 = max_norm(cam2)


			pred1 = torch.squeeze(pred1, -1)
			pred1 = torch.squeeze(pred1, -1)
			pred2 = torch.squeeze(pred2, -1)
			pred2 = torch.squeeze(pred2, -1)
# 			print(pred.size())
			
			# loss
			ce_loss1 = self.loss_ce(pred1, y)
			ce_loss2 = self.loss_ce(pred2, y)

			ec_loss = torch.div(torch.sum(torch.mean(torch.abs(cam1[:,1:,:,:] - cam2[:,1:,:,:])*y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),(2,3))), torch.sum(y))
			# dice_loss = self.loss_dice(pred, y)
			# inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
			ce_loss =  0.5*(ce_loss1 + ce_loss2)
			cur_loss = ce_loss + ec_loss 
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
			iter_accuracy = cls_accuracy(pred1, y)
			# iter_dice_coeff_dice =  Dice(pred, y, self.num_cls)
			# iter_iou = IoU(pred,y, self.num_cls)
			iter_accuracy = iter_accuracy[0].cpu().numpy()
			accuracy.update(iter_accuracy)

			# dice_coeff_hard.update(iter_dice_coeff_dice.cpu().numpy())
			# iou.update(iter_iou.cpu().numpy())

			# pred_max = torch.sigmoid(pred) > 0.5
			# metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())

			self.current_iteration += 1

			if self.current_iteration == 1:
				validation_values = self.validate()
				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("ec_loss",{"train_ec_loss":ec_loss.item()}, self.current_iteration)
				self.summary_writer.add_scalars("accuracy",{"train_accuracy":iter_accuracy, "valid_accuracy":validation_values['accuracy']}
					, self.current_iteration)





			if self.current_iteration%200 == 0:
				validation_values = self.validate()
				print("Epoch-" + str(self.current_epoch))
				print("Train CE loss: " + str(ce_loss.item()))
				print("Validation CE loss: " + str(validation_values['ce_loss']))
				print("Train Accuracy: " + str(iter_accuracy))
				print("Validation Accuracy: " + str(validation_values['accuracy'])+ "\n")

				#Adding Scalars
				self.summary_writer.add_scalars("ce_loss",{"train_ce_loss":ce_loss.item(), "valid_ce_loss":validation_values['ce_loss']}
					, self.current_iteration)
				self.summary_writer.add_scalars("ec_loss",{"train_ec_loss":ec_loss.item()}, self.current_iteration)
				self.summary_writer.add_scalars("accuracy",{"train_accuracy":iter_accuracy, "valid_accuracy":validation_values['accuracy']}
					, self.current_iteration)

				is_best = validation_values['accuracy'] > self.best_metric_value

				if(is_best):
					self.best_metric_value = validation_values['accuracy']
				self.save_checkpoint(is_best=is_best)
# 				self.summary_writer.add_hparams(self.config,{'accuracy':self.best_metric_value})
				
				# self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)



		tqdm_batch.close()
		print("Epoch Average Accuracy- {}".format(accuracy.val))
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
		# dice = AverageMeter()
		ce = AverageMeter()

		# inv_dice = AverageMeter()

		# Initialize average meters of metrics
		accuracy = AverageMeter()
		# dice_coeff_hard = AverageMeterList(self.num_cls)
		# iou = AverageMeterList(self.num_cls)


		#epoch loss
		# metrics = IOUMetric(self.config.num_classes)

		for x, y in tqdm_batch:
			if self.cuda:
				x, y = x.pin_memory().cuda(async=self.config.async_loading), y.cuda(async=self.config.async_loading)
			x, y = Variable(x), Variable(y)


			# model
			pred, cam = self.model(x)
# 			print(pred.size())            
			pred = torch.squeeze(pred, -1)
			pred = torch.squeeze(pred, -1)
# 			print(pred.size()) 
			# loss
			ce_loss = self.loss_ce(pred, y)
			# dice_loss = self.loss_dice(pred, y)
			# inv_dice_loss = self.loss_inv_dice(torch.sigmoid(pred), (y>0.5).float())
			cur_loss =  ce_loss 
#             if np.isnan(float(cur_loss.item())):
#                 raise ValueError('Loss is nan during training...')

			
			ce.update(ce_loss.item())
			# dice.update(dice_loss.item())
			# inv_dice.update(inv_dice_loss.item())


			##calculate accuracy and update its average list
			iter_accuracy = cls_accuracy(pred, y)
			# iter_dice_coeff_dice =  Dice(pred, y, self.num_cls)
			# iter_iou = IoU(pred,y, self.num_cls)
			iter_accuracy = iter_accuracy[0].cpu().numpy()
			accuracy.update(iter_accuracy)

			# dice_coeff_hard.update(iter_dice_coeff_dice.cpu().numpy())
			# iou.update(iter_iou.cpu().numpy())

			# pred_max = torch.sigmoid(pred) > 0.5
			# metrics.add_batch(pred_max.data.cpu().numpy(), y.data.cpu().numpy())



			# 	#Adding Scalars

			# 	# self.summary_writer.add_scalar("epoch_train/dice_loss", iter_accuracy[0], self.current_iteration)


		tqdm_batch.close()
		# print("Epoch Average Accuracy- {}".format(accuracy.val))
		return {'accuracy': accuracy.val,
				'ce_loss': ce.val}


	def test(self):

		self.infer(data='val')
		self.infer(data='test')


	def save_predictions(self, data):


		if self.current_epoch==self.config.max_epoch:



			self.load_checkpoint("model_best.pth.tar")
			self.current_epoch = self.config.max_epoch

			
			project_path = "/home/gp104/projects/def-josedolz/gp104/project_wss_brats/project_wss/"
			dest_tar = os.path.join(project_path, self.config.exp_name, 'predictions_{}.tar'.format(data))
			
			if os.path.isfile(dest_tar):
				print("Predictions already saved!.")
				return

			if data =='val':
				tqdm_batch = tqdm(self.data_loader.validation_infer_loader, total=self.data_loader.validation_infer_iterations,
				desc="Epoch-{}-".format(1))
			elif data =='test':
				tqdm_batch = tqdm(self.data_loader.test_infer_loader, total=self.data_loader.test_infer_iterations,
				desc="Epoch-{}-".format(1))
			else:
				return
			
	# 		project_path = "/home/gp104/projects/def-josedolz/gp104/project_wss_brats/project_wss/"
	# 		dest_tar = os.path.join(project_path, self.config.exp_name, 'predictions_{}.tar'.format(data))
			
	#         if os.path.isfile(dest_tar):
	#             print("Predictions already saved! run test.")
	#             return


			gt_dir = os.path.join(self.config.data_root_infer, "{}/GT/".format(data))
			dest_root_cam = os.path.join(self.config.data_root_infer, "{}/{}/GT_CAM_{}/".format(self.config.exp_name, data, self.config.modality[0]))
			dest_root_gradcam = os.path.join(self.config.data_root_infer, "{}/{}/GT_GRADCAM_{}/".format(self.config.exp_name,data, self.config.modality[0]))
			dest_root_gradcampp = os.path.join(self.config.data_root_infer, "{}/{}/GT_GRADCAMPP_{}/".format(self.config.exp_name, data, self.config.modality[0]))

			
			if not os.path.exists(dest_root_cam):
				os.makedirs(dest_root_cam)
			if not os.path.exists(dest_root_gradcam):
				os.makedirs(dest_root_gradcam)
			if not os.path.exists(dest_root_gradcampp):
				os.makedirs(dest_root_gradcampp)


			self.model.eval()

			cam = CAM(self.model, self.model.backbone)
			# grad_cam_pp = GradCAMpp(self.model, self.model.backbone)
	##############################################################
			for x, y, img_name in tqdm_batch:
				if self.cuda:
					x, y = x.pin_memory().cuda(), y.pin_memory().cuda(non_blocking=True)
				x, y = Variable(x), Variable(y)


				# print(len(img_name))
				cam_map, grad_cam_map, grad_cam_pp_map, logit = cam(x, class_idx=1)

				# logit, cam = self.model(x)

				prob = F.softmax(logit,dim=1)
	#     h_x = prob.squeeze()
				idx = torch.argmax(prob,1)
				# cam = torch.squeeze(cam, 0)
				# pred = torch.squeeze(pred)

				#CAM
				# cam = torch.squeeze(cam, 0) #[2, h, w]
				idx = torch.squeeze(idx, 0).item() #[h, w]
				cam_map, grad_cam_map, grad_cam_pp_map = torch.squeeze(cam_map), torch.squeeze(grad_cam_map), torch.squeeze(grad_cam_pp_map)

				# cam_normed = max_norm(cam) #normalized in the range [0,1]
				save_cam(cam = cam_map, idx = idx, dest_root=dest_root_cam, img_name=img_name[0])
				save_cam(cam = grad_cam_map, idx = idx, dest_root=dest_root_gradcam, img_name=img_name[0])
				save_cam(cam = grad_cam_pp_map, idx = idx, dest_root=dest_root_gradcampp, img_name=img_name[0])
				
				
	# ########################################################################################################################
	# 			#GradCAM
	# 			# saliency_map, _ = grad_cam(input=x, class_idx=1)
			if self.config.run_on_cluster:
				output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
				output_string = output_bytes.decode('utf-8').strip()

			data_path = output_string
			project_path = "/home/gp104/projects/def-josedolz/gp104/project_wss_brats/project_wss/"
			dest_tar = os.path.join(project_path, self.config.exp_name, 'predictions_{}.tar'.format(data))
			dir_to_tar = os.path.join("brats_data", self.config.exp_name, data)
			list_files = subprocess.run(["tar", "-C", data_path, "-cf", dest_tar, dir_to_tar], check=True)
			print("Prediction compression and sacing executed with : %d" % list_files.returncode)

		else:
			print("Skipping predictions saving")
			return
		#tar -C  data_path -cvf projectdir/config.experiment_name/predictions.tar brats_data/config.experiment_name
		
			#
##################################################################################################################

	def infer(self, data):
		gt_dir = os.path.join(self.config.data_root_infer, "{}/GT/".format(data))
		dest_root_cam = os.path.join(self.config.data_root_infer, "{}/{}/GT_CAM_{}/".format(self.config.exp_name, data, self.config.modality[0]))
		dest_root_gradcam = os.path.join(self.config.data_root_infer, "{}/{}/GT_GRADCAM_{}/".format(self.config.exp_name,data, self.config.modality[0]))
		dest_root_gradcampp = os.path.join(self.config.data_root_infer, "{}/{}/GT_GRADCAMPP_{}/".format(self.config.exp_name, data, self.config.modality[0]))        

		if self.config.run_on_cluster:
			output_bytes = subprocess.check_output("echo $SLURM_TMPDIR", shell=True)
			output_string = output_bytes.decode('utf-8').strip()
			
		data_path = output_string
		project_path = "/home/gp104/projects/def-josedolz/gp104/project_wss_brats/project_wss/"
		dest_tar = os.path.join(project_path, self.config.exp_name, 'predictions_{}.tar'.format(data))
# 		dir_to_tar = os.path.join("brats_data", self.config.exp_name)

		#tar xf dest_tar -C data_path
		list_files = subprocess.run(["tar","-xf", dest_tar, "-C", data_path], check=True)
		print("Prediction compression and sacing executed with : %d" % list_files.returncode)
		


		img_names = os.listdir(gt_dir)
		unique_patients = list(set(["_".join(img_name.split("_")[:-1] + [""]) for img_name in img_names]))

		# print(unique_patients)

		thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

		# mean_dice_3d = {0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7:[], 0.8:[]}
		# Initializing empyt list for storing metrics for each patient's scan
		dice_3d = {sal_map: {x: [] for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}
		# dice_3d_cam = {x: [] for x in thresholds}
		# dice_3d_gradcam = {x: [] for x in thresholds}
		# dice_3d_gradcampp = {x: [] for x in thresholds}
		# surface_distance_3d_cam = {x: [] for x in thresholds}
		# surface_distance_3d_gradcam = {x: [] for x in thresholds}
		# surface_distance_3d_gradcampp = {x: [] for x in thresholds}
		surface_distance_3d = {sal_map: {x: [] for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}
		hd_distance_3d = {sal_map: {x: [] for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}
		avd_3d = {sal_map: {x: [] for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}
		spacing = [1.,1.,1.]

		for patient_id in tqdm(unique_patients):
			reconstructed_gt = reconstruct3D(root = gt_dir, patient_id = patient_id)
			reconstructed_cam_pred = reconstruct3D(root = dest_root_cam, patient_id = patient_id)
			reconstructed_gradcam_pred = reconstruct3D(root = dest_root_gradcam, patient_id = patient_id)
			reconstructed_gradcampp_pred = reconstruct3D(root = dest_root_gradcampp, patient_id = patient_id)


			S_cam = reconstructed_cam_pred/255.#reconstructed_pred/255.
			S_gradcam = reconstructed_gradcam_pred/255.
			S_gradcampp = reconstructed_gradcampp_pred/255.
			G = reconstructed_gt>0.5 # tumor_category_values >=1

			pred_map = {'cam' : S_cam, 'grad_cam': S_gradcam, 'grad_cam_pp': S_gradcampp}

			for maps in ['cam', 'grad_cam', 'grad_cam_pp']:
				dsc = list(map(lambda x: dc(pred_map[maps]>x,G), thresholds))
				dice_3d[maps][thresholds[0]].append(dsc[0])
				dice_3d[maps][thresholds[1]].append(dsc[1])
				dice_3d[maps][thresholds[2]].append(dsc[2])
				dice_3d[maps][thresholds[3]].append(dsc[3])
				dice_3d[maps][thresholds[4]].append(dsc[4])
				dice_3d[maps][thresholds[5]].append(dsc[5])
				dice_3d[maps][thresholds[6]].append(dsc[6])

				average_sd = list(map(lambda x: assd(pred_map[maps]>x,G), thresholds))

				surface_distance_3d[maps][thresholds[0]].append(average_sd[0])
				surface_distance_3d[maps][thresholds[1]].append(average_sd[1])
				surface_distance_3d[maps][thresholds[2]].append(average_sd[2])
				surface_distance_3d[maps][thresholds[3]].append(average_sd[3])
				surface_distance_3d[maps][thresholds[4]].append(average_sd[4])
				surface_distance_3d[maps][thresholds[5]].append(average_sd[5])
				surface_distance_3d[maps][thresholds[6]].append(average_sd[6])

				# average_hd = list(map(lambda x: hd(pred_map[maps]>x,G), thresholds))

				# hd_distance_3d[maps][thresholds[0]].append(average_hd[0])
				# hd_distance_3d[maps][thresholds[1]].append(average_hd[1])
				# hd_distance_3d[maps][thresholds[2]].append(average_hd[2])
				# hd_distance_3d[maps][thresholds[3]].append(average_hd[3])
				# hd_distance_3d[maps][thresholds[4]].append(average_hd[4])
				# hd_distance_3d[maps][thresholds[5]].append(average_hd[5])
				# hd_distance_3d[maps][thresholds[6]].append(average_hd[6])

				# average_avd = list(map(lambda x: ravd(pred_map[maps]>x,G), thresholds))

				# avd_3d[maps][thresholds[0]].append(average_avd[0])
				# avd_3d[maps][thresholds[1]].append(average_avd[1])
				# avd_3d[maps][thresholds[2]].append(average_avd[2])
				# avd_3d[maps][thresholds[3]].append(average_avd[3])
				# avd_3d[maps][thresholds[4]].append(average_avd[4])
				# avd_3d[maps][thresholds[5]].append(average_avd[5])
				# avd_3d[maps][thresholds[6]].append(average_avd[6])

				# dice_3d[maps]
				# map(lambda x: dc(pred_map[maps]>x,G), thresholds ) 


# 			for threshold in thresholds:
# 				dice_3d[threshold].append(Dice(S>threshold,G, version='np'))
# 				surface_distance_3d[threshold].append(compute_average_surface_distance_2(compute_surface_distances(G, S>threshold, spacing)))

		mean_dice_3d =  {sal_map : {x : (np.mean(np.array(dice_3d[sal_map][x])), np.std(np.array(dice_3d[sal_map][x]))) for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}   
		dice_dest = os.path.join(self.config.project_directory, self.config.exp_name, "{}_dice_3d.json".format(data))
		with open(dice_dest, 'w') as fp:
			json.dump(mean_dice_3d, fp)
			
		mean_surface_distance_3d = {sal_map : {x : (np.mean(np.array(surface_distance_3d[sal_map][x])),np.std(np.array(surface_distance_3d[sal_map][x]))) for x in thresholds} for sal_map in ['cam', 'grad_cam', 'grad_cam_pp']}
		sd_dest = os.path.join(self.config.project_directory, self.config.exp_name, "{}_sd_3d.json".format(data))
		with open(sd_dest, 'w') as fp:
			json.dump(mean_surface_distance_3d, fp)
			
		return



# 		classes = {0: 'non-tumor', 1: 'tumor'}
# 		root = os.path.join("../../brats_classification/","train/MR_{}".format(config.modality))
# # gt_root = os.path.join(config.data_root, "val/GT")
# 		dest_root = "../../brats_classification/train/GT_CAM_{}/".format(config.modality)
# 		image_names = os.listdir(root)
# 		image_names.sort()
# 		for image_name in tqdm(image_names[:200]):
# 		    image_path = os.path.join(root,image_name)
# 		    img = Image.open(image_path)
# #     feature_blobs.clear()
# 		    get_cam(model, img, image_name, image_path, dest_root,get_heat=True)

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

