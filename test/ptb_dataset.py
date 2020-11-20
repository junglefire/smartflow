#!/usr/bin/env python
# -*- coding:utf-8 -*-
from smartflow.dataset import ptb
from smartflow.optimizer import *
from smartflow.trainer import *
from smartflow.config import *
from smartflow.layers import *
from smartflow.util import *

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

ptb = ptb.PTB("./dataset/ptb")
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