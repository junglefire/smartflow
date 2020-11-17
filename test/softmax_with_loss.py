#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from smartflow.layers import *
import numpy as np

def softmax(x):
	if x.ndim == 2:
		x = x - x.max(axis=1, keepdims=True)
		x = np.exp(x)
		x /= x.sum(axis=1, keepdims=True)
	elif x.ndim == 1:
		x = x - np.max(x)
		x = np.exp(x) / np.sum(np.exp(x))
	return x

def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	if t.size == y.size:
		t = t.argmax(axis=1)
	batch_size = y.shape[0]
	print("y=", y)
	print("y=", np.arange(batch_size))
	print("y=", y[np.arange(batch_size), t])
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
	def __init__(self):
		self.params, self.grads = [], []
		self.y = None  
		self.t = None  

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		if self.t.size == self.y.size:
			self.t = self.t.argmax(axis=1)
		loss = cross_entropy_error(self.y, self.t)
		return loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = self.y.copy()
		dx[np.arange(batch_size), self.t] -= 1
		dx *= dout
		dx = dx / batch_size
		return dx

softmaxWithLoss = SoftmaxWithLoss()

x = np.array([
	[0.3, 0.2, 0.5],
	[0.1, 0.2, 0.7]
])
t = np.array([
	[0, 1, 0],
	[0, 0, 1]
])
out = softmaxWithLoss.forward(x, t)
print(out)

dx = softmaxWithLoss.backward(1)
print(dx)

