#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging as log
import urllib.request
import numpy as np
import os.path
import gzip
import pickle
import os

# download `mnist` dataset
class Mnist:
	def __init__(self, dataset_dir):
		self.__url_base = 'http://yann.lecun.com/exdb/mnist/'
		self.__key_file = {
			'train_img': 'train-images-idx3-ubyte.gz',
			'train_label': 'train-labels-idx1-ubyte.gz',
			'test_img': 't10k-images-idx3-ubyte.gz',
			'test_label': 't10k-labels-idx1-ubyte.gz'
		}
		self.__dataset_dir = dataset_dir
		self.__save_file = self.__dataset_dir + "/mnist.pkl"
		self.train_num = 60000
		self.test_num = 10000
		self.img_dim = (1, 28, 28)
		self.img_size = 784
		pass

	def __del__(self):
		pass

	def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
		# if the pickle file not exist
		if not os.path.exists(self.__save_file):
			# download the mnist dataset files, unzip the files and load into numpy array
			# persist the array to the pickle file
			self.__init_mnist()
		# load the pickle file
		log.info("load pickle file...")
		with open(self.__save_file, 'rb') as f:
			dataset = pickle.load(f)
		# normalize the dataset
		if normalize:
			for key in ('train_img', 'test_img'):
				dataset[key] = dataset[key].astype(np.float32)
				dataset[key] /= 255.0
		# encode label
		if one_hot_label:
			dataset['train_label'] = self.__change_one_hot_label(dataset['train_label'])
			dataset['test_label'] = self.__change_one_hot_label(dataset['test_label'])
		# flat the dataset to 1-d array
		if not flatten:
			for key in ('train_img', 'test_img'):
				dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
		log.info("done!")
		return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

	def __init_mnist(self):
		log.info("download `mnist` dataset ...")
		self.__download_mnist()
		dataset = self.__convert_numpy()
		log.info("creating pickle file ...")
		with open(self.__save_file, 'wb') as f:
			pickle.dump(dataset, f, -1)
		log.info("init mnist done!")

	def __download_mnist(self):
		for v in self.__key_file.values():
			file_path = self.__dataset_dir + "/" + v
			if os.path.exists(file_path):
				return
			log.info("downloading file `%s` ..." % v)
			urllib.request.urlretrieve(self.__url_base + v, file_path)

	def __convert_numpy(self):
		dataset = {}
		dataset['train_img'] =  self.__load_img(self.__key_file['train_img'])
		dataset['train_label'] = self.__load_label(self.__key_file['train_label'])
		dataset['test_img'] = self.__load_img(self.__key_file['test_img'])
		dataset['test_label'] = self.__load_label(self.__key_file['test_label'])
		return dataset

	def __load_img(self, file_name):
		file_path = self.__dataset_dir + "/" + file_name
		log.info("converting image file `%s` to NumPy Array ... " % file_name)
		with gzip.open(file_path, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		data = data.reshape(-1, self.img_size)
		return data

	def __load_label(self, file_name):
		file_path = self.__dataset_dir  + "/" + file_name
		log.info("converting label file `%s` to NumPy Array ..." % file_name)
		with gzip.open(file_path, 'rb') as f:
			labels = np.frombuffer(f.read(), np.uint8, offset=8)
		return labels

	def __change_one_hot_label(self, X):
		T = np.zeros((X.size, 10))
		for idx, row in enumerate(T):
			row[X[idx]] = 1
		return T

if __name__ == '__main__':
	from PIL import Image
	log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.DEBUG)
	mnist = Mnist("./mnist")
	((x_train, t_train), (x_test, t_test)) = mnist.load_mnist(normalize=False, flatten=True, one_hot_label=False)
	log.info("train dataset shape : %s" % str(x_train.shape))
	log.info("train label shape   : %s" % str(t_train.shape))
	log.info("test dataset shape  : %s" % str(x_test.shape))
	log.info("test label shape    : %s" % str(t_test.shape))
	# image show
	log.info("show image-0...")
	img = x_train[0] 
	label = t_train[0]
	log.info("label of image-0 is `%s`" % label)
	img = img.reshape(28, 28)
	pil_img = Image.fromarray(np.uint8(img)) 
	pil_img.show()