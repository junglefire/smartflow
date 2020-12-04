# coding: utf-8
import sys
sys.path.append('..')
from ..base.function import *
from ..base.layers import *
from ..base.np import *

# 这里定义的2层神经网络只用于说明NN的基本原理，不能在实际应用中使用
class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)
		return y
		
	def loss(self, x, t):
		y = self.predict(x)
		return cross_entropy_error(y, t)
	
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy
	
	# 使用数值法计算梯度	
	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		return grads
		
	def gradient(self, x, t):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		grads = {}
		batch_num = x.shape[0]
		# forward
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)
		# backward
		dy = (y - t) / batch_num
		grads['W2'] = np.dot(z1.T, dy)
		grads['b2'] = np.sum(dy, axis=0)
		dz1 = np.dot(dy, W2.T)
		da1 = sigmoid_grad(a1) * dz1
		grads['W1'] = np.dot(x.T, da1)
		grads['b1'] = np.sum(da1, axis=0)
		return grads

# 使用Layer构建两层NN
class TwoLayerNetEx:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size
		# 参数初始化
		W1 = 0.01 * np.random.randn(I, H)
		b1 = np.zeros(H)
		W2 = 0.01 * np.random.randn(H, O)
		b2 = np.zeros(O)
		# 构建层
		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2)
		]
		self.loss_layer = SoftmaxWithLoss()
		# 存储参数和对应的梯度
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def forward(self, x, t):
		score = self.predict(x)
		loss = self.loss_layer.forward(score, t)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout
