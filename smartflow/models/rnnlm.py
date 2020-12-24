# coding: utf-8
import sys
sys.path.append('..')
from ..base.time_layers import *
from ..base.base_model import *
from ..base.function import *
from ..base.layers import *
from ..base.np import *

#
# 一个简单的RNN语言模型，基于RNN和TimeRNN类
class SimpleRnnlm:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		# V表示词库的大小 / D表示词向量的大小 / H表示隐藏层的大小
		V, D, H = vocab_size, wordvec_size, hidden_size
		# 初始化权重
		# RNN层和Affine层使用了`Xavier初始值`
		embed_W = (np.random.randn(V, D) / 100).astype('f')
		rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
		rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
		rnn_b = np.zeros(H).astype('f')
		affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		# 搭建层
		self.layers = [
			TimeEmbedding(embed_W),
			TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.rnn_layer = self.layers[1]
		# 存储参数
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def forward(self, xs, ts):
		for layer in self.layers:
			xs = layer.forward(xs)
		loss = self.loss_layer.forward(xs, ts)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		self.rnn_layer.reset_state()


#
# 基于LSTM和TimeLSTM类的RNN语言模型
class Rnnlm(BaseModel):
	def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
		# V表示词库的大小 / D表示词向量的大小 / H表示隐藏层的大小
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		# 初始化权重
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		# 生成模型
		self.layers = [
			TimeEmbedding(embed_W),
			TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
			TimeAffine(affine_W, affine_b)
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.lstm_layer = self.layers[1]
		# すべての重みと勾配をリストにまとめる
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, xs):
		for layer in self.layers:
			xs = layer.forward(xs)
		return xs

	def forward(self, xs, ts):
		score = self.predict(xs)
		loss = self.loss_layer.forward(score, ts)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		self.lstm_layer.reset_state()


#
# LSTM模型的优化，基于以下论文：
#  [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
#  [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
#  [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
class BetterRnnlm(BaseModel):
	def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		# 权重初始化
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b1 = np.zeros(4 * H).astype('f')
		lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b2 = np.zeros(4 * H).astype('f')
		affine_b = np.zeros(V).astype('f')
		# 构建模型
		self.layers = [
			TimeEmbedding(embed_W),
			TimeDropout(dropout_ratio),
			TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
			TimeDropout(dropout_ratio),
			TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
			TimeDropout(dropout_ratio),
			TimeAffine(embed_W.T, affine_b)  # weight tying!!
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.lstm_layers = [self.layers[2], self.layers[4]]
		self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, xs, train_flg=False):
		for layer in self.drop_layers:
			layer.train_flg = train_flg
		for layer in self.layers:
			xs = layer.forward(xs)
		return xs

	def forward(self, xs, ts, train_flg=True):
		score = self.predict(xs, train_flg)
		loss = self.loss_layer.forward(score, ts)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		for layer in self.lstm_layers:
			layer.reset_state()


#
# 基于单层LSTM的文本生成类
class RnnlmGen(Rnnlm):
	def generate(self, start_id, skip_ids=None, sample_size=100):
		word_ids = [start_id]
		x = start_id
		while len(word_ids) < sample_size:
			x = np.array(x).reshape(1, 1)
			score = self.predict(x)
			p = softmax(score.flatten())
			# 随机选择目标词
			sampled = np.random.choice(len(p), size=1, p=p)
			if (skip_ids is None) or (sampled not in skip_ids):
				x = sampled
				word_ids.append(int(x))
		return word_ids

	def get_state(self):
		return self.lstm_layer.h, self.lstm_layer.c

	def set_state(self, state):
		self.lstm_layer.set_state(*state)


#
# 基于多层LSTM的文本生成类
class BetterRnnlmGen(BetterRnnlm):
	def generate(self, start_id, skip_ids=None, sample_size=100):
		word_ids = [start_id]
		x = start_id
		while len(word_ids) < sample_size:
			x = np.array(x).reshape(1, 1)
			score = self.predict(x).flatten()
			p = softmax(score).flatten()
			# 按概率提取目标词
			sampled = np.random.choice(len(p), size=1, p=p)
			if (skip_ids is None) or (sampled not in skip_ids):
				x = sampled
				word_ids.append(int(x))
		return word_ids

	def get_state(self):
		states = []
		for layer in self.lstm_layers:
			states.append((layer.h, layer.c))
		return states

	def set_state(self, states):
		for layer, state in zip(self.lstm_layers, states):
			layer.set_state(*state)



