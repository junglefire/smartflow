# coding: utf-8
import sys
sys.path.append('..')
from ..base.attention_layer import *
from ..base.time_layers import *
from ..base.base_model import *
from ..base.function import *
from ..base.layers import *
from ..base.np import *

#
# 编码器
class Encoder:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		# 初始化参数
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		# 构建模型
		self.embed = TimeEmbedding(embed_W)
		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
		# 存储参数
		self.params = self.embed.params + self.lstm.params
		self.grads = self.embed.grads + self.lstm.grads
		self.hs = None

	def forward(self, xs):
		xs = self.embed.forward(xs)
		hs = self.lstm.forward(xs)
		self.hs = hs
		return hs[:, -1, :]

	def backward(self, dh):
		dhs = np.zeros_like(self.hs)
		dhs[:, -1, :] = dh
		dout = self.lstm.backward(dhs)
		dout = self.embed.backward(dout)
		return dout


#
# 解码器
class Decoder:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		# 初始化参数
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		# 构建模型
		self.embed = TimeEmbedding(embed_W)
		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
		self.affine = TimeAffine(affine_W, affine_b)
		# 存储参数
		self.params, self.grads = [], []
		for layer in (self.embed, self.lstm, self.affine):
			self.params += layer.params
			self.grads += layer.grads

	def forward(self, xs, h):
		self.lstm.set_state(h)
		out = self.embed.forward(xs)
		out = self.lstm.forward(out)
		score = self.affine.forward(out)
		return score

	def backward(self, dscore):
		dout = self.affine.backward(dscore)
		dout = self.lstm.backward(dout)
		dout = self.embed.backward(dout)
		dh = self.lstm.dh
		return dh

	def generate(self, h, start_id, sample_size):
		sampled = []
		sample_id = start_id
		self.lstm.set_state(h)
		# 生成目标序列
		for _ in range(sample_size):
			x = np.array(sample_id).reshape((1, 1))
			out = self.embed.forward(x)
			out = self.lstm.forward(out)
			score = self.affine.forward(out)
			sample_id = np.argmax(score.flatten())
			sampled.append(int(sample_id))
		return sampled


#
# seq2seq模型
class Seq2seq(BaseModel):
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		self.encoder = Encoder(V, D, H)
		self.decoder = Decoder(V, D, H)
		self.softmax = TimeSoftmaxWithLoss()

		self.params = self.encoder.params + self.decoder.params
		self.grads = self.encoder.grads + self.decoder.grads

	def forward(self, xs, ts):
		decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

		h = self.encoder.forward(xs)
		score = self.decoder.forward(decoder_xs, h)
		loss = self.softmax.forward(score, decoder_ts)
		return loss

	def backward(self, dout=1):
		dout = self.softmax.backward(dout)
		dh = self.decoder.backward(dout)
		dout = self.encoder.backward(dh)
		return dout

	def generate(self, xs, start_id, sample_size):
		h = self.encoder.forward(xs)
		sampled = self.decoder.generate(h, start_id, sample_size)
		return sampled


#
# 偷窥(Peeky)解码器
class PeekyDecoder:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		self.embed = TimeEmbedding(embed_W)
		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
		self.affine = TimeAffine(affine_W, affine_b)
		self.params, self.grads = [], []
		for layer in (self.embed, self.lstm, self.affine):
			self.params += layer.params
			self.grads += layer.grads
		self.cache = None

	def forward(self, xs, h):
		N, T = xs.shape
		N, H = h.shape
		self.lstm.set_state(h)
		out = self.embed.forward(xs)
		hs = np.repeat(h, T, axis=0).reshape(N, T, H)
		out = np.concatenate((hs, out), axis=2)
		out = self.lstm.forward(out)
		out = np.concatenate((hs, out), axis=2)
		score = self.affine.forward(out)
		self.cache = H
		return score

	def backward(self, dscore):
		H = self.cache
		dout = self.affine.backward(dscore)
		dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
		dout = self.lstm.backward(dout)
		dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
		self.embed.backward(dembed)
		dhs = dhs0 + dhs1
		dh = self.lstm.dh + np.sum(dhs, axis=1)
		return dh

	def generate(self, h, start_id, sample_size):
		sampled = []
		char_id = start_id
		self.lstm.set_state(h)
		H = h.shape[1]
		peeky_h = h.reshape(1, 1, H)
		for _ in range(sample_size):
			x = np.array([char_id]).reshape((1, 1))
			out = self.embed.forward(x)
			out = np.concatenate((peeky_h, out), axis=2)
			out = self.lstm.forward(out)
			out = np.concatenate((peeky_h, out), axis=2)
			score = self.affine.forward(out)
			char_id = np.argmax(score.flatten())
			sampled.append(char_id)
		return sampled


#
# 基于偷窥解码器的Seq2Seq网络模型
class PeekySeq2seq(Seq2seq):
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		self.encoder = Encoder(V, D, H)
		self.decoder = PeekyDecoder(V, D, H)
		self.softmax = TimeSoftmaxWithLoss()
		self.params = self.encoder.params + self.decoder.params
		self.grads = self.encoder.grads + self.decoder.grads


#
# 基于Attention的编码器实现
class AttentionEncoder(Encoder):
	def forward(self, xs):
		xs = self.embed.forward(xs)
		hs = self.lstm.forward(xs)
		return hs

	def backward(self, dhs):
		dout = self.lstm.backward(dhs)
		dout = self.embed.backward(dout)
		return dout


#
# 基于Attention的解码器实现
class AttentionDecoder:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
		affine_b = np.zeros(V).astype('f')
		self.embed = TimeEmbedding(embed_W)
		self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
		self.attention = TimeAttention()
		self.affine = TimeAffine(affine_W, affine_b)
		layers = [self.embed, self.lstm, self.attention, self.affine]
		self.params, self.grads = [], []
		for layer in layers:
			self.params += layer.params
			self.grads += layer.grads

	def forward(self, xs, enc_hs):
		h = enc_hs[:,-1]
		self.lstm.set_state(h)
		out = self.embed.forward(xs)
		dec_hs = self.lstm.forward(out)
		c = self.attention.forward(enc_hs, dec_hs)
		out = np.concatenate((c, dec_hs), axis=2)
		score = self.affine.forward(out)
		return score

	def backward(self, dscore):
		dout = self.affine.backward(dscore)
		N, T, H2 = dout.shape
		H = H2 // 2
		dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]
		denc_hs, ddec_hs1 = self.attention.backward(dc)
		ddec_hs = ddec_hs0 + ddec_hs1
		dout = self.lstm.backward(ddec_hs)
		dh = self.lstm.dh
		denc_hs[:, -1] += dh
		self.embed.backward(dout)
		return denc_hs

	def generate(self, enc_hs, start_id, sample_size):
		sampled = []
		sample_id = start_id
		h = enc_hs[:, -1]
		self.lstm.set_state(h)
		for _ in range(sample_size):
			x = np.array([sample_id]).reshape((1, 1))
			out = self.embed.forward(x)
			dec_hs = self.lstm.forward(out)
			c = self.attention.forward(enc_hs, dec_hs)
			out = np.concatenate((c, dec_hs), axis=2)
			score = self.affine.forward(out)
			sample_id = np.argmax(score.flatten())
			sampled.append(sample_id)
		return sampled


#
# 基于Attention的seq2seq实现
class AttentionSeq2seq(Seq2seq):
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		args = vocab_size, wordvec_size, hidden_size
		self.encoder = AttentionEncoder(*args)
		self.decoder = AttentionDecoder(*args)
		self.softmax = TimeSoftmaxWithLoss()
		self.params = self.encoder.params + self.decoder.params
		self.grads = self.encoder.grads + self.decoder.grads




