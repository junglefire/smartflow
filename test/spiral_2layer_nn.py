#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import logging as log
import sys

from smartflow.dataset import spiral
from smartflow.optimizer import *
from smartflow.trainer import *
from smartflow.config import *
from smartflow.layers import *

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

#
# 显示数据集
def __show_dataset():
	x, t = spiral.load_data() 
	log.info('x.shape: %s', x.shape) # (300, 2) 
	log.info('t.shape: %s', t.shape) # (300, 3)
	# 绘图
	N = 100
	CLS_NUM = 3
	markers = ['o', 'x', '^']
	for i in range(CLS_NUM):
		plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
	plt.show()

#
# 自定义train流程
def __custom_train():
	max_epoch = 300
	batch_size = 30
	hidden_size = 10
	learning_rate = 1.0
	# 加载数据
	x, t = spiral.load_data()
	model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
	optimizer = SGD(lr=learning_rate)
	# 学習で使用する変数
	data_size = len(x)
	max_iters = data_size // batch_size
	total_loss = 0
	loss_count = 0
	loss_list = []
	# 训练
	for epoch in range(max_epoch):
		idx = np.random.permutation(data_size)
		x = x[idx]
		t = t[idx]
		# epoch loop
		for iters in range(max_iters):
			batch_x = x[iters*batch_size:(iters+1)*batch_size]
			batch_t = t[iters*batch_size:(iters+1)*batch_size]
			loss = model.forward(batch_x, batch_t)
			model.backward()
			optimizer.update(model.params, model.grads)
			# 计算损失
			total_loss += loss
			loss_count += 1
			# 定期的に学習経過を出力
			if (iters+1) % 10 == 0:
				avg_loss = total_loss / loss_count
				print('| epoch %d |  iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
				loss_list.append(avg_loss)
				total_loss, loss_count = 0, 0
	# 显示学习结果
	plt.plot(np.arange(len(loss_list)), loss_list, label='train')
	plt.xlabel('iterations (x10)')
	plt.ylabel('loss')
	plt.show()

#
# 使用训练器
def __exec_trainer():
	max_epoch = 300
	batch_size = 30
	hidden_size = 10
	learning_rate = 1.0
	# 生成数据集
	x, t = spiral.load_data()
	model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
	optimizer = SGD(lr=learning_rate)
	trainer = Trainer(model, optimizer)
	trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
	trainer.plot()


#
# 定义两层神经网络
class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size
		# 初始化
		W1 = 0.01 * np.random.randn(I, H)
		b1 = np.zeros(H)
		W2 = 0.01 * np.random.randn(H, O)
		b2 = np.zeros(O)
		# 定义NN的层
		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2)
		]
		self.loss_layer = SoftmaxWithLoss()
		# 存储参数和梯度
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

#
# 主进程
if __name__ == "__main__":
	# 命令行参数  :
	# 	- show  : 显示数据集图形
	#	- train : 用神经网络学习数据集
	if len(sys.argv) != 2:
		log.info("[Usage]spiral_2layer_nn.py <show>|<train>")
		sys.exit()
	if (sys.argv[1] == 'show'):
		__show_dataset()
	elif (sys.argv[1] == 'train'):
		# __custom_train()
		__exec_trainer()
	else:
		log.info("invalid command, abort!")
