#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import smartflow as sf
import logging as log
import numpy as np
import shutil
import pickle
import click
import time
import sys
import os

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

@click.command()
@click.option('--reverse', default=True, flag_value='reverse', help='reverse input data')
def train(reverse):
	# 加载数据集
	seq = sf.dataset.sequence.Sequence("./dataset/date.txt")
	(x_train, t_train), (x_test, t_test) = seq.load_data()
	(char_to_id, id_to_char) = seq.get_vocab()
	# 反转输入语句
	# 是否反转输入数据集
	if reverse:
		x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
	# 设定超参数 
	vocab_size = len(char_to_id) 
	wordvec_size = 16 
	hidden_size = 256 
	batch_size = 128 
	max_epoch = 10 
	max_grad = 5.0
	# 创建模型
	model = sf.models.seq2seq.AttentionSeq2seq(vocab_size, wordvec_size, hidden_size) 
	optimizer = sf.base.optimizer.Adam() 
	trainer = sf.base.trainer.Trainer(model, optimizer)
	# 存储过程数据
	acc_list = [] 
	# 开始训练
	for epoch in range(max_epoch):
		log.info("epoch#%s..." % (epoch))
		trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
		correct_num = 0 
		for i in range(len(x_test)):
			question, correct = x_test[[i]], t_test[[i]]
			verbose = i < 10 
			correct_num += sf.base.util.eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse=True)
		acc = float(correct_num) / len(x_test) 
		acc_list.append(acc) 
		print('val acc %.3f%%' % (acc * 100))
	# 存储模型参数
	model.save_params("./models/attention_seq2seq.pkl")
	# 显示训练过程准确率的变化曲线
	x = np.arange(len(acc_list))
	plt.plot(x, acc_list, marker='o')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.ylim(0, 1.0)
	plt.show()

if __name__ == "__main__":
	train()



