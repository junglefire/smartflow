# coding: utf-8
from .config import *

def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return np.maximum(0, x)

# tanh函数 
def tanh(x):
	return np.tanh(x)


