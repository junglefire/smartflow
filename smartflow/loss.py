# coding: utf-8
from .config import *

# softmax函数的输出是0.0到1.0之间的实数，softmax函数的输出值的总和是1
# 因此,可以把softmax函数的输出解释为`概率`
def softmax(x):
	if x.ndim == 2:
		x = x - x.max(axis=1, keepdims=True)
		x = np.exp(x)
		x /= x.sum(axis=1, keepdims=True)
	elif x.ndim == 1:
		# 防溢出対策
		x = x - np.max(x)
		x = np.exp(x) / np.sum(np.exp(x))
	return x

# 交叉熵误差
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	if t.size == y.size:
		t = t.argmax(axis=1)
	batch_size = y.shape[0]
	# `np.arange(batch_size)`会生成一个从0到`batch_size-1`的数组。比如当`batch_size`为5
	# 时，`np.arange(batch_size)`会生成一个`NumPy`数组`[0, 1, 2, 3, 4]`。因为t中标签是以 
	# `[2, 7, 0, 9, 4]`的形式存储的，所以`y[np.arange(batch_size), t]`能抽出各个数据的正确
	# 解标签对应的神经网络的输出(即生成`NumPy`数组`[y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]`)
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
