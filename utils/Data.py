# -*- coding: utf-8 -*-
# # @Author  : jiabing

#百度知道的数据集暂时没去处理，有点问题

import pandas as pd
import jieba
import time
import json
import re


# # 问答数据构造
# start_time = time.time()
#
# data =[]
# i = 0
# for line in open("../data/zhidao_qa.json", encoding='UTF-8'):
#     if i < 100000: # 只用了前10万个条数据
#         data_list = json.loads(line)
#         # print(data_list.keys())
#         # print(data_list.values())
#         #data.append([data_list["answers"][0],data_list["question"],data_list["tags"][:]])
#         data.append([ data_list["question"],data_list["answers"][0]])
#         i +=1
# # print(data)
# df = pd.DataFrame(data, columns=['knowledge_q', 'knowledge_a'])
# df.to_csv("../data/train_qa.csv",index=False)
# #print(df)



# 问答数据导入
data = pd.read_csv("../data/train_qa.csv")
data = data[:5000]
# # drop null值
# data.dropna(subset=['knowledge_q','knowledge_a'],inplace=True)
# data.to_csv('../data/train_qa.csv')


# 分词、去停用词
def sentence_process(sen):
    """
    对句子进行处理, 以字符串形式进行无用词处理, 以列表形式对停用词处理, 再组合为句子
    :param sen: 需要处理的句子
    :return: 处理后的句子
    """
    # 处理无用词
    sen = clean_sentence(sen)
    # 切词
    sen = jieba.cut(sen)
    # 过滤停用词
    stop_word = read_stop_words('../data/stopwords/stop_words.txt')
    data = [i for i in sen if i not in stop_word]
    # 去除频率较低的词 TODO

    return ' '.join(data)

def clean_sentence(data):
    """
    把句子中无用词替换为空
    :param data:
    :return:
    """

    return re.sub(r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”、~@#￥%……&*（）a-zA-Z0]',
                          '', data) if isinstance(data, str) else ''


def read_stop_words(stop_word_path):
    """
    读取停用词表
    :param stop_word_path: 停用词表的路径
    :return: ->list
    """
    with open(stop_word_path, encoding='utf-8') as f:
        return [word.strip() for word in f.readlines()]

data['knowledge_q'] = data['knowledge_q'].apply(lambda x: sentence_process(x))
data['knowledge_a'] = data['knowledge_a'].apply(lambda x: sentence_process(x))

# 保存一下分好词的训练集
data.to_csv('../data/train_segment.csv')

# 为了word2vec词向量训练, 生成一句一句的切好词的文本
data['Merge'] = data[['knowledge_q','knowledge_a']].apply(lambda x: ' '.join(x), axis=1)
merge_seg_data = data['Merge']

# 保存一下合并的分好词的句子文件,用于word2vec的训练
merge_seg_data.to_csv("../data/merge_segment.csv", header=None, index=None)

