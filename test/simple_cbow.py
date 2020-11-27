#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from smartflow.util import preprocess, create_contexts_target, convert_one_hot
# from smartflow.models import simple_word2vec as sw
# from smartflow.trainer import Trainer
# from smartflow.optimizer import Adam

import smartflow as sf

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = sf.base.util.preprocess(text)

vocab_size = len(word_to_id)
contexts, target = sf.base.util.create_contexts_target(corpus, window_size)
target = sf.base.util.convert_one_hot(target, vocab_size)
contexts = sf.base.util.convert_one_hot(contexts, vocab_size)

model = sf.models.simple_word2vec.SimpleCBOW(vocab_size, hidden_size)
optimizer = sf.base.optimizer.Adam()
trainer = sf.base.trainer.Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])