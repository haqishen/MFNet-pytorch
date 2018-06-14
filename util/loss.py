# coding:utf-8
import torch.nn.functional as F

def softmax_with_cross_entropy(logits, target, weight=None):
	probs = F.softmax(logits, dim=1)
	loss  = F.cross_entropy(probs, target, weight=weight)
	return loss