#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import smartflow as sf
import numpy as np 

#用subplot()方法绘制多幅图形
plt.figure(figsize=(10, 10),dpi=80)

ax1 = plt.subplot(221)
x = np.arange(-5.0, 5.0, 0.1)
y = sf.base.function.step_function(x) 
plt.plot(x, y, color="r", linestyle="--") 
plt.ylim(-0.1, 1.1) # 指定y轴的范围 
plt.title("step function")

ax2 = plt.subplot(222)
x = np.arange(-5.0, 5.0, 0.1)
y = sf.base.function.sigmoid(x) 
plt.plot(x, y, color="b", linestyle="-") 
plt.ylim(-0.1, 1.1)
plt.title("sigmoid function")

ax3 = plt.subplot(223)
x = np.arange(-3.0, 15.0, 0.1)
y = sf.base.function.relu(x) 
plt.plot(x, y) 
plt.ylim(-0.1, 15) # 指定y轴的范围 
plt.title("relu function")

ax4 = plt.subplot(224)
x = np.arange(-5.0, 5.0, 0.1)
y = sf.base.function.tanh(x) 
plt.plot(x, y) 
plt.ylim(-1.1, 1.1) # 指定y轴的范围 
plt.title("tanh function")

plt.show()