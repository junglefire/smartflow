#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from smartflow.util import preprocess, create_contexts_target, convert_one_hot
# from smartflow.models import simple_word2vec as sw
# from smartflow.trainer import Trainer
# from smartflow.optimizer import Adam

import smartflow as sf
import numpy as np
import pickle

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 加载PTB数据
ptb = sf.dataset.ptb.PTB("./dataset/ptb")
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = sf.base.util.create_contexts_target(corpus, window_size)
if sf.base.config.GPU:
	contexts, target = to_gpu(contexts), to_gpu(target)

# 创建CBOW模型
# model = sf.models.word2vec.CBOW(vocab_size, hidden_size, window_size, corpus)
model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = sf.base.optimizer.Adam()
trainer = sf.base.trainer.Trainer(model, optimizer)

# 学習開始
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 保存必要数据，以便后续使用
word_vecs = model.word_vecs
if sf.base.config.GPU:
	word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  
# or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
	pickle.dump(params, f, -1)