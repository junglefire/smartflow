#!/usr/bin/env python
# -*- coding:utf-8 -*-
import smartflow as sf
import logging as log
import numpy as np
import sys

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

text = 'You say goodbye and I say hello.'

# 文本向量化
corpus, word_to_id, id_to_word = sf.base.util.preprocess(text) 

# 生成共现矩阵
vocab_size = len(word_to_id)
C = sf.base.util.create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']] 	# you的单词向量
c1 = C[word_to_id['i']]		# i的单词向量 
log.info("co-occurrence matrix is: %s", C)
log.info("cos similarity is: %s", sf.base.util.cos_similarity(c0, c1))

# 取相似度最高的5个单词
sf.base.util.most_similar('you', word_to_id, id_to_word, C, top=5)

# 计算PPMI
C = sf.base.util.create_co_matrix(corpus, vocab_size) 
W = sf.base.util.ppmi(C)
np.set_printoptions(precision=3) # 有效位数为3位
log.info('PPMI: \n%s', W)

# 基于SVD建模
# U, S, V = np.linalg.svd(W)

