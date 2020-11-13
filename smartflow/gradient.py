# coding: utf-8
from .activator import *
from .config import *

# 使用数值计算方法计算函数的梯度
def numerical_gradient(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)
		x[idx] = tmp_val - h 
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val
		it.iternext()   
	return grad

# sigmoid函数的梯度
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)