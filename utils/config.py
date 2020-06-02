# -*- coding: utf-8 -*-
# @Author  : jiabing
import os
import pathlib

# 获取根目录
root = pathlib.Path(__file__).parent.parent

# 训练集目录
TRAIN_PATH = os.path.join(root, 'data', 'train_qa.csv')

# 停用词表
STOP_WORD_PATH = os.path.join(root, 'data', 'stopwords/stop_words.txt')

# 词向量模型保存路径
WV_MODEL_PATH = os.path.join(root, 'data', 'word2vec', 'word2vec.model')

# 词向量矩阵保存路径
EMBEDDING_MATRIX_PATH = os.path.join(root, 'data', 'word2vec', 'embedding_matrix')

# 字典保存路径
SAVE_VOCAB_PATH = os.path.join(root, 'data', 'word2vec', 'vocab.txt')
SAVE_REVERSE_PATH = os.path.join(root, 'data', 'word2vec', 'reverse_vocab.txt')

