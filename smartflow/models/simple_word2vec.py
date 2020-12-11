#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append('..')
from ..base.layers import *
import numpy as np

#
# 一个简版的CBOW模型，仅用于学习，参考原书3.2节
class SimpleCBOW:
	def __init__(self, vocab_size, hidden_size):
		V, H = vocab_size, hidden_size
		W_in = 0.01 * np.random.randn(V, H).astype('f')
		W_out = 0.01 * np.random.randn(H, V).astype('f')
		self.in_layer0 = MatMulLayer(W_in)
		self.in_layer1 = MatMulLayer(W_in)
		self.out_layer = MatMulLayer(W_out)
		self.loss_layer = SoftmaxWithLoss()
		layers = [self.in_layer0, self.in_layer1, self.out_layer]
		self.params, self.grads = [], []
		for layer in layers:
			self.params += layer.params
			self.grads += layer.grads
		self.word_vecs = W_in

	def forward(self, contexts, target):
		h0 = self.in_layer0.forward(contexts[:, 0])
		h1 = self.in_layer1.forward(contexts[:, 1])
		h = (h0 + h1) * 0.5
		score = self.out_layer.forward(h)
		loss = self.loss_layer.forward(score, target)
		return loss

	def backward(self, dout=1):
		ds = self.loss_layer.backward(dout)
		da = self.out_layer.backward(ds)
		da *= 0.5
		self.in_layer1.backward(da)
		self.in_layer0.backward(da)
		return None

#
# 一个简版的SkipGram模型，仅用于学习
class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        self.in_layer = MatMulLayer(W_in)
        self.out_layer = MatMulLayer(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
