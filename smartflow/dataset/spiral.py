#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging as log
import numpy as np
import os

# 生成双螺旋数据集
def load_data(seed=1984):
	np.random.seed(seed)
	# 数据集每个簇包含的数据记录条数
	N = 100  
	# 维度
	DIM = 2
	# 簇的个数  
	CLS_NUM = 3  
	# x是二维数据
	x = np.zeros((N*CLS_NUM, DIM))
	# t是onehot编码的标签数据
	t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int)
	# 生成数据
	for j in range(CLS_NUM):
		for i in range(N):
			rate = i/N
			radius = 1.0*rate
			theta = j*4.0 + 4.0*rate + np.random.randn()*0.2
			ix = N*j + i
			x[ix] = np.array([radius*np.sin(theta), radius*np.cos(theta)]).flatten()
			t[ix, j] = 1
	return x, t

if __name__ == "__main__":
	# 导入图形库
	import matplotlib.pyplot as plt
	# 打开日志开关
	log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)
	# 生成测试数据
	x, t = load_data() 
	log.info('x.shape: %s', x.shape) # (300, 2) 
	log.info('t.shape: %s', t.shape) # (300, 3)
	# 绘图
	N = 100
	CLS_NUM = 3
	markers = ['o', 'x', '^']
	for i in range(CLS_NUM):
		plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
	plt.show()