import torch    
import torch.nn as nn


def L2_norm(p,q, label_mask, eps):
	loss = torch.mean(torch.pow((p/torch.norm(p+eps, dim=(-1,-2,), keepdim=True)) - (q/torch.norm(q+eps, dim=(-1,-2,), keepdim=True)),2)*label_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
	return loss

class MapLossL2Norm(nn.Module):
	def __init__(self, epsilon=1e-12):
		super(MapLossL2Norm, self).__init__()
		self.epsilon = epsilon

	def forward(self, p, q, label_mask):

		return L2_norm(p,q,label_mask, epsilon)