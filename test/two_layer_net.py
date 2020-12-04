#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import smartflow as sf
import logging as log
import numpy as np

# from smartflow.dataset import mnist
# from smartflow import simple_nn
# from smartflow.config import *

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

mnist = sf.dataset.mnist.Mnist("dataset")
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(one_hot_label=True)

network = sf.models.simple_nn.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

iters_num = 10000 
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
	log.info("epoch-#%d..." % i)
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	# 计算梯度
	# 数值法会非常的慢
	# grad = network.numerical_gradient(x_batch, t_batch)
	grad = network.gradient(x_batch, t_batch)
	# 参数更新
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]
   	# 计算损失函数 
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)
	# 保存精度
	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 显示精度变化曲线
markers = {'train': 'o', 'test': 's'}

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

