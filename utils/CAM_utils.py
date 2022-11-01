import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os




class CAM:
	"""
		CAM object to compute, CAM, GradCAM, GradCAM++

	"""
	def __init__(self, arch, target_layer):
		self.model_arch = arch

		self.gradients = dict()
		self.activations = dict()

		def backward_hook(module, grad_input, grad_output):
			self.gradients['value'] = grad_output[0]

		def forward_hook(module, input, output):
			self.activations['value'] = output

		target_layer.register_forward_hook(forward_hook)
		target_layer.register_backward_hook(backward_hook)


	def saliency_map_size(self, *input_size):
		device = next(self.model_arch.parameters()).device
		self.model_arch(torch.zeros(1, 3, *input_size, device=device))
		return self.activations['value'].shape[2:]

	def gradcam(self, gradients, activations,gradients_size, saliency_map_size):
		e=1e-8
		h, w = saliency_map_size
		b, k, u, v = gradients_size
		alpha = gradients.view(b, k, -1).mean(2)
		# alpha = F.relu(gradienif self.config.mode == 'train':
		weights = alpha.view(b, k, 1, 1)

		saliency_map = (weights*activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=True)
		# saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		# saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
		max_v = torch.max(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		min_v = torch.min(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		saliency_map = F.relu(saliency_map-min_v-e)/(max_v-min_v+e)

		return saliency_map

	def gradcampp(self, gradients, activations, score,gradients_size, saliency_map_size):
		e=1e-8
		h, w = saliency_map_size
		b, k, u, v = gradients_size
		alpha_num = gradients.pow(2)
		alpha_denom = alpha_num.mul(2) + activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
		alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

		alpha = alpha_num.div(alpha_denom+1e-7)
		positive_gradients = F.relu(score.exp()*gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
		weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

		saliency_map = (weights*activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=True)
		# saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		# saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

		max_v = torch.max(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		min_v = torch.min(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		saliency_map = F.relu(saliency_map-min_v-e)/(max_v-min_v+e)

		return saliency_map

		# return saliency_map

	def cam(self, cam, saliency_map_size):
		e=1e-8
		h, w = saliency_map_size
		saliency_map = torch.unsqueeze(cam, 1)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=True)
		# saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		# saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data
		max_v = torch.max(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		min_v = torch.min(saliency_map.view(1,-1),dim=-1)[0].view(1,1)
		saliency_map = F.relu(saliency_map-min_v-e)/(max_v-min_v+e)

		# return saliency_map

		return saliency_map



	def forward(self, input, class_idx=None, retain_graph=False):
		b, c, h, w = input.size()

		logit, cam = self.model_arch(input)
		if class_idx is None:
			score = logit[:, logit.max(1)[-1]].squeeze()
			cam = cam[:, logit.max(1)[-1]].squeeze()
		else:
			score = logit[:, class_idx].squeeze()
			cam = cam[:, class_idx]

		self.model_arch.zero_grad()
		score.backward(retain_graph=retain_graph)
		gradients = self.gradients['value']
		activations = self.activations['value']
		b, k, u, v = gradients.size()

		grad_cam_map = self.gradcam(gradients, activations, gradients_size = (b, k, u, v), saliency_map_size = (h,w))
		grad_campp_map = self.gradcampp(gradients, activations, score, gradients_size =(b, k ,u, v), saliency_map_size = (h, w))
		cam_map = self.cam(cam, saliency_map_size = (h,w))

		return cam_map, grad_cam_map, grad_campp_map, logit

	def __call__(self, input, class_idx=None, retain_graph=False):
		return self.forward(input, class_idx, retain_graph)


def max_norm(p, version='torch', e=1e-8):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p


def save_cam(cam, idx, dest_root, img_name, target_idx=1):
	
	CAM = cam.cpu().detach().numpy()
	CAM_tumor = CAM * idx 
	cam_img = np.uint8(255. * CAM_tumor)
	img = Image.fromarray(cam_img).convert('L')
	img.save(os.path.join(dest_root,img_name))
	return
