#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import smartflow as sf
import logging as log
import numpy as np
import shutil
import pickle
import time
import sys
import os

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

# 设定超参数
batch_size = 5
wordvec_size = 100

# RNN的隐藏状态向量的元素个数
hidden_size = 100

# Truncated BPTT的时间跨度大小
time_size = 5

# 生成[0-(n-1)]作为测试集，标签集则是[1-(n)]
corpus_size = 100
corpus = np.arange(0, corpus_size)
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

# 输入: 取语料库的[0:-1]的元素作为训练数据
xs = corpus[:-1]
# 输出(监督标签): 取每个元素的后一个元素作为标签 
ts = corpus[1:] 
# 数据集大小
data_size = len(xs) 

log.info('corpus size: %d, vocabulary size: %d， dataset size: %d' % (corpus_size, vocab_size, data_size))

# 最大循环数表示要多少个循环能把整个数据集的元素都取一遍
max_iters = data_size // (batch_size * time_size)

log.info('max_iters is: %d' % (max_iters))

# 计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

log.info('jump step: %d, offsets: %s, time_size: %d' % (jump, offsets, time_size))

# 生成模型
model = sf.models.rnnlm.SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = sf.base.optimizer.SGD(0.1)

# 训练需要的参数
max_epoch = 100

total_loss = 0
loss_count = 0
ppl_list = []

time_idx = 0

# 循环训练 
for epoch in range(max_epoch):
	for iter in range(max_iters):
		# 获取mini-batch
		batch_x = np.empty((batch_size, time_size), dtype='i')
		batch_t = np.empty((batch_size, time_size), dtype='i')
		for t in range(time_size):
			for i, offset in enumerate(offsets):
				# log.info('i: %d, t: %d, offset: %d' % (i, t, offset))
				# log.info('pos: %d' % ((offset + time_idx) % data_size))
				batch_x[i, t] = xs[(offset + time_idx) % data_size]
				batch_t[i, t] = ts[(offset + time_idx) % data_size]
			time_idx += 1
		log.info("#epoch-%d, batch_x is:\n%s" % (iter, batch_x))
		# 计算梯度，更新参数
		loss = model.forward(batch_x, batch_t)
		model.backward()
		optimizer.update(model.params, model.grads)
		total_loss += loss
		loss_count += 1

	# 各个epoch的困惑度评价
	ppl = np.exp(total_loss / loss_count)
	log.info('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
	ppl_list.append(float(ppl))
	total_loss, loss_count = 0, 0
	break

# 画图
"""
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
"""




