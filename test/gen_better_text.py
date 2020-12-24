#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import smartflow as sf
import logging as log
import numpy as np
import shutil
import pickle
import time
import sys
import os

log.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=log.INFO)

# 读入训练数据
ptb = sf.dataset.ptb.PTB("./dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

# 构建模型、加载参数
model = sf.models.rnnlm.BetterRnnlmGen()
model.load_params('./models/ptb_better_lstm.pkl')

# 从一个词开始生成句子
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 文章生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
log.info("start word: %s..." % (start_word))
print(txt)

# 模型重置
model.reset_state()

# 从一组词开始生成句子
start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print('-' * 50)
log.info("start words: %s..." % (start_words))
print(txt)

