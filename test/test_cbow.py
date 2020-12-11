#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from smartflow.util import preprocess, create_contexts_target, convert_one_hot
# from smartflow.models import simple_word2vec as sw
# from smartflow.trainer import Trainer
# from smartflow.optimizer import Adam

import smartflow as sf
import logging as log
import numpy as np
import shutil
import pickle
import time
import sys
import os

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

window_size = 5
hidden_size = 100
batch_size = 100
# batch_size = 500
max_epoch = 10

def __exec_trainer():
	# 备份参数文件
	params_file = "./models/cbow_params.pkl"
	if os.path.exists(params_file):
		bak_file = params_file + "." + time.strftime("%Y%m%d.%H%M%S", time.localtime())
		shutil.move(params_file, bak_file)
	# 加载PTB数据
	ptb = sf.dataset.ptb.PTB("./dataset/ptb")
	corpus, word_to_id, id_to_word = ptb.load_data('train')
	vocab_size = len(word_to_id)
	# 生成源和目标
	contexts, target = sf.base.util.create_contexts_target(corpus, window_size)
	if sf.base.config.GPU:
		contexts, target = to_gpu(contexts), to_gpu(target)
	# 创建CBOW模型
	model = sf.models.word2vec.CBOW(vocab_size, hidden_size, window_size, corpus)
	# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
	optimizer = sf.base.optimizer.Adam()
	trainer = sf.base.trainer.Trainer(model, optimizer)
	# 学習開始
	trainer.fit(contexts, target, max_epoch, batch_size)
	# 保存必要数据，以便后续使用
	word_vecs = model.word_vecs
	if sf.base.config.GPU:
		word_vecs = to_cpu(word_vecs)
	params = {}
	params['word_vecs'] = word_vecs.astype(np.float16)
	params['word_to_id'] = word_to_id
	params['id_to_word'] = id_to_word
	# or 'skipgram_params.pkl'
	with open(params_file, 'wb') as f:
		pickle.dump(params, f, -1)
	# 显示训练趋势图
	trainer.plot()

def __exec_eval():
	params_file = "./models/cbow_params.pkl"
	with open(params_file, 'rb') as f: 
		params = pickle.load(f)
		word_vecs = params['word_vecs']
		word_to_id = params['word_to_id']
		id_to_word = params['id_to_word']
	# 计算相似度
	querys = ['you', 'year', 'car', 'toyota']
	for query in querys:
		sf.base.util.most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

#
# 主进程
if __name__ == "__main__":
	# 命令行参数  :
	# 	- eval  : 显示数据集图形
	#	- train : 用神经网络学习数据集
	if len(sys.argv) != 2:
		log.info("[Usage]test_cbow.py <eval>|<train>")
		sys.exit()
	if (sys.argv[1] == 'eval'):
		__exec_eval()
	elif (sys.argv[1] == 'train'):
		# __custom_train()
		__exec_trainer()
	else:
		log.info("invalid command, abort!")