#!/usr/bin/env python
# -*- coding:utf-8 -*-
from .layers import Embedding, SigmoidWithLoss
from .config import GPU
from .function import *
from .np import * 
import collections

#
# 优化中间层到输出层的计算性能
class EmbeddingDot:
	def __init__(self, W):
		self.embed = Embedding(W)
		self.params = self.embed.params
		self.grads = self.embed.grads
		self.cache = None

	def forward(self, h, idx):
		target_W = self.embed.forward(idx)
		out = np.sum(target_W * h, axis=1)
		self.cache = (h, target_W)
		return out

	def backward(self, dout):
		h, target_W = self.cache
		dout = dout.reshape(dout.shape[0], 1)
		dtarget_W = dout * h
		self.embed.backward(dtarget_W)
		dh = dout * target_W
		return dh

#
# 基于语料库中各个单词的出现次数求出概率分布后，根据这个概率分布进行采样
class UnigramSampler:
	def __init__(self, corpus, power, sample_size):
		self.sample_size = sample_size
		self.vocab_size = None
		self.word_p = None
		# 计算每个单词出现的次数
		counts = collections.Counter()
		for word_id in corpus:
			counts[word_id] += 1
		# 词库的大小
		vocab_size = len(counts)
		self.vocab_size = vocab_size
		# 计算每个单词出现的概率
		self.word_p = np.zeros(vocab_size)
		for i in range(vocab_size):
			self.word_p[i] = counts[i]
		# 对原来的概率分布取`power`次方，通常为0.75，目的是为了防止低频单词被忽略
		# 更准确地说，通过取0.75次方，低频单词的概率将稍微变高。
		self.word_p = np.power(self.word_p, power)
		self.word_p /= np.sum(self.word_p)

	# 取负样例
	# target表示正例在词库中的索引
	def get_negative_sample(self, target):
		batch_size = target.shape[0]
		if not GPU:
			negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
			for i in range(batch_size):
				p = self.word_p.copy()
				target_idx = target[i]
				p[target_idx] = 0
				p /= p.sum()
				negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
		else:
			# GPU(cupy）
			negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
											   replace=True, p=self.word_p)
		return negative_sample

#
# 负采样的损失函数
class NegativeSamplingLoss:
	def __init__(self, W, corpus, power=0.75, sample_size=5):
		self.sample_size = sample_size
		self.sampler = UnigramSampler(corpus, power, sample_size)
		# 生成`sample_size+1`个层，这是因为需要生成一个正例用的层和`sample_size`个负例用的层
		self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
		self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
		self.params, self.grads = [], []
		for layer in self.embed_dot_layers:
			self.params += layer.params
			self.grads += layer.grads

	def forward(self, h, target):
		batch_size = target.shape[0]
		negative_sample = self.sampler.get_negative_sample(target)
		# 正例损失
		score = self.embed_dot_layers[0].forward(h, target)
		correct_label = np.ones(batch_size, dtype=np.int32)
		loss = self.loss_layers[0].forward(score, correct_label)
		# 负例损失
		negative_label = np.zeros(batch_size, dtype=np.int32)
		for i in range(self.sample_size):
			negative_target = negative_sample[:, i]
			score = self.embed_dot_layers[1 + i].forward(h, negative_target)
			loss += self.loss_layers[1 + i].forward(score, negative_label)
		return loss

	def backward(self, dout=1):
		dh = 0
		for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
			dscore = l0.backward(dout)
			dh += l1.backward(dscore)
		return dh

