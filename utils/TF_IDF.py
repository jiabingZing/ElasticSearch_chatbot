# -*- coding: utf-8 -*-
# # @Author  : jiabing
from gensim import corpora, models, similarities
import jieba
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
data = pd.read_csv("../data/train_segment.csv")
# train = data["knowledge_q"][0]

# # 生成corpus用于tf_idf训练（sklearn 版，简单但不怎么好用）
# corpus=[]
# for i in range(len(data)):
#     corpus.append(data["knowledge_q"][i])
# corpus=[str(tmp) for tmp in corpus]
# tfidf_vec = TfidfVectorizer()
# tfidf_matrix = tfidf_vec.fit_transform(corpus)

# # 得到语料库所有不重复的词
# print(tfidf_vec.get_feature_names())
#
# # 得到每个单词对应的id值
# print(tfidf_vec.vocabulary_)
#
# # 得到每个句子的向量
# # 向量里数字的顺序是按照词语的id顺序来的
# print(tfidf_matrix.toarray())


# 生成texts用于tf_idf训练(gensim 方法)
train_set=[]
train_set.append(word for word in data["knowledge_q"])
documents = train_set[0]
texts = [[word for word in str(document).split()] for document in train_set[0]]
# print(texts) [['房地产', '开发', '经营', '企业', '开发', '产品', '利息支出', '可否', '税前', '扣除'],['国家税务总局', '增值税', '若干', '税收政策', '批复'],...,]

# 计算词频
frequency = defaultdict(int)  # 构建一个字典对象
# 遍历分词后的结果集，计算每个词出现的频率
for text in texts:
    for token in text:
        frequency[token] += 1
# 选择频率大于1的词
texts = [[token for token in text if frequency[token] > 1] for text in texts]

# 创建字典（单词与编号之间的映射）
dictionary = corpora.Dictionary(texts)
# print(dictionary.token2id) #{'产品': 0,'企业': 1,'利息支出': 2,'可否': 3,'开发': 4,'房地产': 5,...}

# 建立语料库,将每一篇文档转换为向量
# doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
corpus = [dictionary.doc2bow(text) for text in texts]# # [[[(0, 1), (1, 1), (2, 1)], [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],..]

# 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
# tfidf = models.TfidfModel(corpus)

# 相识度计算
def semblance(text, corpus):
    # 对测试文本分词
    dic_text_list = list(jieba.cut(text))

    # 制作测试文本的词袋
    doc_text_vec = dictionary.doc2bow(dic_text_list)

    # 获取语料库每个文档中每个词的tfidf值，即用tfidf模型训练语料库
    tfidf = models.TfidfModel(corpus)

    # 对稀疏向量建立索引
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sim = index[tfidf[doc_text_vec]]  # 相当于sim = index.get_similarities(tfidf[doc_text_vec])
    # 按照相似度来排序
    sim_sorted = sorted(enumerate(sim, 1), key=lambda x: -x[1])
    # 相当于sorted(enumerate(sim), key=lambda x: x[1], reverse=True
    print(sim_sorted)
    return sim_sorted
# es所有相似度匹配和分词都是用本身数据库的，Es比tfidf速度要快
# 但ES的相似度评估是不受控制的，因为不能传入我们的算法
# 两种算法能不能做个整合：第一步用ES求topn的相似度返回，在把topn放在tfidf中
# 计算top n的相似度，相当于前面返回的相似度用tfidf做个筛选，
# 这样tfidf他的相似度计算减少了计算量，Es的相似度计算可以直接用tfidf相似度方法计算

if __name__ == '__main__':
    text = '房地产开发经营企业开发产品利息支出可否税前扣除'
    semblance(text, corpus) #选取了第一条数据做测试



