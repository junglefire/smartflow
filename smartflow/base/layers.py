# coding: utf-8
from .np import *  # import numpy as np
from .config import GPU
from .function import *

# 
# 加法层
class AddLayer:
	def __init__(self):
		pass

	def forward(self, x, y):
		out = x + y
		return out

	def backward(self, dout):
		dx = dout * 1
		dy = dout * 1
		print(type(dx))
		return dx, dy

#
# 乘法层
class MulLayer:
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		self.x = x
		self.y = y				
		out = x * y

		return out

	def backward(self, dout):
		dx = dout * self.y
		dy = dout * self.x

		return dx, dy


#
# 矩阵乘法层
class MatMulLayer:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.x = None

	def forward(self, x):
		W, = self.params
		out = np.dot(x, W)
		self.x = x
		return out

	def backward(self, dout):
		W, = self.params
		dx = np.dot(dout, W.T)
		dW = np.dot(self.x.T, dout)
		self.grads[0][...] = dW
		return dx

#
# ReLU层
class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0
		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout
		return dx


#
# Sigmoid层
class Sigmoid:
	def __init__(self):
		self.params, self.grads = [], []
		self.out = None

	def forward(self, x: np.ndarray)-> np.ndarray:
		out = 1 / (1 + np.exp(-x))
		self.out = out
		return out

	def backward(self, dout: np.ndarray)-> np.ndarray:
		dx = dout * (1.0 - self.out) * self.out
		return dx


#
# 仿射变换层
class Affine:
	def __init__(self, W: np.ndarray, b: np.ndarray):
		self.params = [W, b]
		self.grads = [np.zeros_like(W), np.zeros_like(b)]
		self.x = None

	def forward(self, x: np.ndarray)-> np.ndarray:
		W, b = self.params
		out = np.dot(x, W) + b
		self.x = x
		return out

	def backward(self, dout: np.ndarray)-> np.ndarray:
		W, b = self.params
		dx = np.dot(dout, W.T)
		dW = np.dot(self.x.T, dout)
		db = np.sum(dout, axis=0)
		self.grads[0][...] = dW
		self.grads[1][...] = db
		return dx


#
# Softmax层
class Softmax:
	def __init__(self):
		self.params, self.grads = [], []
		self.out = None

	def forward(self, x: np.ndarray)-> np.ndarray:
		self.out = softmax(x)
		return self.out

	def backward(self, dout: np.ndarray)-> np.ndarray:
		dx = self.out * dout
		sumdx = np.sum(dx, axis=1, keepdims=True)
		dx -= self.out * sumdx
		return dx


#
# SoftmaxWithLoss层
class SoftmaxWithLoss:
	def __init__(self):
		self.params, self.grads = [], []
		self.y = None  
		self.t = None  

	def forward(self, x: np.ndarray, t: np.ndarray)-> np.float64:
		self.t = t
		self.y = softmax(x)
		if self.t.size == self.y.size:
			self.t = self.t.argmax(axis=1)
		loss = cross_entropy_error(self.y, self.t)
		return loss

	def backward(self, dout: np.float64=1.0)-> np.ndarray:
		batch_size = self.t.shape[0]
		dx = self.y.copy()
		dx[np.arange(batch_size), self.t] -= 1
		dx *= dout
		dx = dx / batch_size
		return dx


#
# SigmoidWithLoss层
class SigmoidWithLoss:
	def __init__(self):
		self.params, self.grads = [], []
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = 1 / (1 + np.exp(-x))
		self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) * dout / batch_size
		return dx


class Dropout:
	'''
	http://arxiv.org/abs/1207.0580
	'''
	def __init__(self, dropout_ratio=0.5):
		self.params, self.grads = [], []
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			return x * self.mask
		else:
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		return dout * self.mask