#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging as log
import urllib.request
import numpy as np
import os.path
import pickle
import gzip
import os

# download `mnist` dataset
class PTB:
	def __init__(self, dataset_dir):
		# self.url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
		self.url_base = 'https://github.com/wojzaremba/lstm/blob/master/data/'
		self.key_file = {
			'train':'ptb.train.txt',
			'test':'ptb.test.txt',
			'valid':'ptb.valid.txt'
		}
		self.save_file = {
			'train':'ptb.train.npy',
			'test':'ptb.test.npy',
			'valid':'ptb.valid.npy'
		}
		self.vocab_file = 'ptb.vocab.pkl'
		self.dataset_dir = dataset_dir
		pass

	def __del__(self):
		pass

	# datatype: 'train'/'test'/'valid'
	# 返回值：
	#  - corpus中保存了单词ID列表
	#  - id_to_word是将单词ID转化为单词的字典
	#  - word_to_id是将单词转化为单词ID的字典
	def load_data(self, data_type='train'):
		if data_type not in ['train', 'test', 'valid']:
			raise Exception('data_type must be `train` or `test` or `valid`')
		# 数据存储目录
		save_path = self.dataset_dir + '/' + self.save_file[data_type]
		# 加载词典
		word_to_id, id_to_word = self.load_vocab()
		# 加载已存档的语料库
		if os.path.exists(save_path):
			corpus = np.load(save_path)
			return corpus, word_to_id, id_to_word
		# 如果没有存档文件，则重新创建语料库并存档
		file_name = self.key_file[data_type]
		file_path = self.dataset_dir + '/' + file_name
		self.__download(file_name)
		# 将换行用`<eos>`代替
		words = open(file_path).read().replace('\n', '<eos>').strip().split()
		corpus = np.array([word_to_id[w] for w in words])
		np.save(save_path, corpus)
		return corpus, word_to_id, id_to_word

	# 加载词库
	def load_vocab(self):
		vocab_path = self.dataset_dir + '/' + self.vocab_file
		# 如果词库已经存在则直接加载到内存	
		if os.path.exists(vocab_path):
			with open(vocab_path, 'rb') as f:
				word_to_id, id_to_word = pickle.load(f)
			return word_to_id, id_to_word
		# 创建词库
		word_to_id = {}
		id_to_word = {}
		data_type = 'train'
		file_name = self.key_file[data_type]
		file_path = self.dataset_dir + '/' + file_name
		# 下载词库	
		self.__download(file_name)
		# 将换行用`<eos>`代替
		words = open(file_path).read().replace('\n', '<eos>').strip().split()
		log.info("create vocabulary ...")
		# 向量化	
		for i, word in enumerate(words):
			if word not in word_to_id:
				tmp_id = len(word_to_id)
				word_to_id[word] = tmp_id
				id_to_word[tmp_id] = word
		# 导出向量化结果
		with open(vocab_path, 'wb') as f:
			pickle.dump((word_to_id, id_to_word), f)
		return word_to_id, id_to_word

	def __download(self, file_name):
		file_path = self.dataset_dir + '/' + file_name
		if os.path.exists(file_path):
			return
		log.info('downloading %s ...', self.url_base + file_name)
		try:
			urllib.request.urlretrieve(self.url_base + file_name, file_path)
		except urllib.error.URLError:
			import ssl
			ssl._create_default_https_context = ssl._create_unverified_context
			urllib.request.urlretrieve(self.url_base + file_name, file_path)

#
# main loop
if __name__ == '__main__':
	from PIL import Image
	log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.DEBUG)
	mnist = Mnist("./mnist")
	((x_train, t_train), (x_test, t_test)) = mnist.load_mnist(normalize=False, flatten=True, one_hot_label=False)
	log.info("train dataset shape : %s" % str(x_train.shape))
	log.info("train label shape   : %s" % str(t_train.shape))
	log.info("test dataset shape  : %s" % str(x_test.shape))
	log.info("test label shape	: %s" % str(t_test.shape))
	# image show
	log.info("show image-0...")
	img = x_train[0] 
	label = t_train[0]
	log.info("label of image-0 is `%s`" % label)
	img = img.reshape(28, 28)
	pil_img = Image.fromarray(np.uint8(img)) 
	pil_img.show()