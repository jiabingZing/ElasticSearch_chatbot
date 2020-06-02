# -*- coding: utf-8 -*-
# # @Author  : jiabing
from utils.config import *
import numpy as np
from gensim.models.word2vec import Word2Vec, LineSentence

merge_segment = '../data/merge_segment.csv'
# 训练词向量
print('start build w2v model')
model = Word2Vec(LineSentence(merge_segment), workers=8, negative=5, min_count=5, size=300, window=3, iter=2)
# vocab = model.wv.vocab

# 保存词向量模型
model.save(WV_MODEL_PATH)

# 保存词向量
np.savetxt(EMBEDDING_MATRIX_PATH, model.wv.vectors, fmt='%0.8f')

# 建立字典
vocab = {word: index for index, word in enumerate(model.wv.index2word)}
reverse_vocab = {index: word for index, word in enumerate(model.wv.index2word)}

def save_dict(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        for i, j in data.items():
            f.write('{}\t{}\n'.format(i, j))
save_dict(SAVE_VOCAB_PATH, vocab)
save_dict(SAVE_REVERSE_PATH, vocab)

if __name__ == '__main__':
    # 加载模型
    model = Word2Vec.load(WV_MODEL_PATH)
    embedding_matrix = model.wv.vectors
    print(embedding_matrix.shape)
    print(model["企业"])
