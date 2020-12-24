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
batch_size = 20
wordvec_size = 100
# RNN的隐藏状态向量的元素个数
hidden_size = 100  
# Truncated BPTT的时间跨度大小
time_size = 35 
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 读入训练数据（缩小了数据集） 
ptb = sf.dataset.ptb.PTB("./dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = sf.models.rnnlm.Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = sf.base.optimizer.SGD(lr)
trainer = sf.base.trainer.RnnlmTrainer(model, optimizer)

# 这里不对每个epoch做eval了 
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# テストデータで評価
model.reset_state()
ppl_test = sf.base.util.eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# パラメータの保存
model.save_params("./models/ptb_lstm.pkl")



