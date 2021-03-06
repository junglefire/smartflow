#!/usr/bin/env python
# -*- coding:utf-8 -*-
#from smartflow.dataset import ptb
#from smartflow.optimizer import *
#from smartflow.trainer import *
#from smartflow.config import *
#from smartflow.layers import *
#from smartflow.util import *

import smartflow as sf
import logging as log

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

ptb = sf.dataset.ptb.PTB("./dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data('train')

print('corpus size:', len(corpus)) 
print('corpus[:30]:', corpus[:30]) 
print() 
print('id_to_word[0]:', id_to_word[0]) 
print('id_to_word[1]:', id_to_word[1]) 
print('id_to_word[2]:', id_to_word[2]) 
print() 
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy']) 
print("word_to_id['lexus']:", word_to_id['lexus'])

window_size = 2 
wordvec_size = 100
vocab_size = len(word_to_id) 
print('counting co-occurrence ...')
C = sf.base.util.create_co_matrix(corpus, vocab_size, window_size) 
print('calculating PPMI ...')
W = sf.base.util.ppmi(C, verbose=True)
print('calculating SVD ...')

U, S, V = np.linalg.svd(W)
word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']

for query in querys:
	sf.base.util.most_similar(query, word_to_id, id_to_word, word_vecs, top=5)