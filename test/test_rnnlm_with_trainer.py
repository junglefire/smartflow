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
batch_size = 10
wordvec_size = 100

# RNN的隐藏状态向量的元素个数
hidden_size = 100

# Truncated BPTT的时间跨度大小
time_size = 5
lr = 0.1
max_epoch = 100

# 读入训练数据（缩小了数据集） 
ptb = sf.dataset.ptb.PTB("./dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

# 输入: 取语料库的[0:-1]的元素作为训练数据
xs = corpus[:-1]
# 输出(监督标签): 取每个元素的后一个元素作为标签 
ts = corpus[1:] 
# 数据集大小
data_size = len(xs) 

log.info('corpus size: %d, vocabulary size: %d， dataset size: %d' % (corpus_size, vocab_size, data_size))

# 学习用的参数
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

log.info('max_iters is: %d' % (max_iters))

# 创建模型
model = sf.models.rnnlm.SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = sf.base.optimizer.SGD(lr)
trainer = sf.base.trainer.RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()



