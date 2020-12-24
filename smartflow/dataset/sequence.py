#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
import sys
import os

# load `addition` sequence dataset
class Sequence:
	def __init__(self, filename, seed=1984):
		self.id_to_char = {}
		self.char_to_id = {}
		self.seed = seed
		self.filename = filename

	def __del__(self):
		pass

	def load_data(self):
		if not os.path.exists(self.filename):
			print('No file: %s' % file_name)
			return None
		# 分别存储问题和答案
		questions, answers = [], []
		for line in open(self.filename, 'r'):
			idx = line.find('_')
			questions.append(line[:idx])
			answers.append(line[idx:-1])
		# create vocab dict
		for i in range(len(questions)):
			q, a = questions[i], answers[i]
			self._update_vocab(q)
			self._update_vocab(a)
		# create numpy array
		x = numpy.zeros((len(questions), len(questions[0])), dtype=numpy.int)
		t = numpy.zeros((len(questions), len(answers[0])), dtype=numpy.int)
		for i, sentence in enumerate(questions):
			x[i] = [self.char_to_id[c] for c in list(sentence)]
		for i, sentence in enumerate(answers):
			t[i] = [self.char_to_id[c] for c in list(sentence)]
		# shuffle
		indices = numpy.arange(len(x))
		if self.seed is not None:
			numpy.random.seed(self.seed)
		numpy.random.shuffle(indices)
		x = x[indices]
		t = t[indices]
		# 10% for validation set
		split_at = len(x) - len(x) // 10
		(x_train, x_test) = x[:split_at], x[split_at:]
		(t_train, t_test) = t[:split_at], t[split_at:]
		return (x_train, t_train), (x_test, t_test)

	def _update_vocab(self, txt):
		chars = list(txt)
		for i, char in enumerate(chars):
			if char not in self.char_to_id:
				tmp_id = len(self.char_to_id)
				self.char_to_id[char] = tmp_id
				self.id_to_char[tmp_id] = char

	def get_vocab(self):
		return self.char_to_id, self.id_to_char

#
# main loop
if __name__ == '__main__':
	seq = Sequence(sys.argv[1])
	(x_train, t_train), (x_test, t_test) = seq.load_data()
	print(x_train.shape, t_train.shape) 
	print(x_test.shape, t_test.shape)
	(char_to_id, id_to_char) = seq.get_vocab()
	print(''.join([id_to_char[c] for c in x_train[0]])) 
	print(''.join([id_to_char[c] for c in t_train[0]]))


