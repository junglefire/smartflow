#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append('..')
from ..base.neg_sampling_layer import *
from ..base.layers import *
import numpy as np

#
# 优化版的CBOW模型
class CBOW:
	def __init__(self, vocab_size, hidden_size, window_size, corpus):
		V, H = vocab_size, hidden_size
		W_in = 0.01 * np.random.randn(V, H).astype('f')
		W_out = 0.01 * np.random.randn(V, H).astype('f')
		# 根据窗口大小生成输入层
		self.in_layers = []
		for i in range(2 * window_size):
			layer = Embedding(W_in) 
			self.in_layers.append(layer)
		# 负采样
		self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
		layers = self.in_layers + [self.ns_loss]
		self.params, self.grads = [], []
		for layer in layers:
			self.params += layer.params
			self.grads += layer.grads
		self.word_vecs = W_in

	def forward(self, contexts, target):
		h = 0
		for i, layer in enumerate(self.in_layers):
			h += layer.forward(contexts[:, i])
		h *= 1 / len(self.in_layers)
		loss = self.ns_loss.forward(h, target)
		return loss

	def backward(self, dout=1):
		dout = self.ns_loss.backward(dout)
		dout *= 1 / len(self.in_layers)
		for layer in self.in_layers:
			layer.backward(dout)
		return None

#
# 优化版的SkipGram模型
class SkipGram:
	def __init__(self, vocab_size, hidden_size, window_size, corpus):
		V, H = vocab_size, hidden_size
		rn = np.random.randn
		W_in = 0.01 * rn(V, H).astype('f')
		W_out = 0.01 * rn(V, H).astype('f')
		self.in_layer = Embedding(W_in)
		self.loss_layers = []
		for i in range(2 * window_size):
			layer = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
			self.loss_layers.append(layer)
		layers = [self.in_layer] + self.loss_layers
		self.params, self.grads = [], []
		for layer in layers:
			self.params += layer.params
			self.grads += layer.grads
		self.word_vecs = W_in

	def forward(self, contexts, target):
		h = self.in_layer.forward(target)
		loss = 0
		for i, layer in enumerate(self.loss_layers):
			loss += layer.forward(h, contexts[:, i])
		return loss

	def backward(self, dout=1):
		dh = 0
		for i, layer in enumerate(self.loss_layers):
			dh += layer.backward(dout)
		self.in_layer.backward(dh)
		return None