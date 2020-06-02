# -*- coding: utf-8 -*-
# # @Author  : jiabing

import string
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from gensim.models.word2vec import Word2Vec, LineSentence
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from utils.config import *
class ESearch(object):
    def __init__(self, index_name, doc_type_name, \
            stop_words_file=STOP_WORD_PATH, model_name='es', es_weight=0.2):
        '''
            测试ElasticSearch版本：6.2.4
        '''
        # 导入用户自定义停用词
        self.stop_list = set()  # 停用词储存器
        self.load_stop_words_from_file(stop_words_file)  # 从文件加载停用词并更新到stop_list
        self.word2vec = Word2Vec.load(WV_MODEL_PATH)  # 导入词向量
        self.model_name = model_name
        self.index_name = index_name
        self.doc_type_name = doc_type_name
        self.es = Elasticsearch()
        self.es_status = False  # ES状态
        self.check_es_status()  # 检查index和doc_type
        self.questions_hash = set()
        self.punctuation = set(string.punctuation)  # 标点符号

        # 超参数，打分函数中es和tfidf,word2vec,bert等各自的贡献程度
        self.es_weight = es_weight
        self.bert_weight = 1 - es_weight

    def load_stop_words_from_file(self, file):
        if file:
            with open(file, encoding='utf-8') as f:
                lines = f.readlines()
            self.stop_list.update([x.strip() for x in lines])

    def check_es_status(self):
        print('==========')
        if self.check_exist_index(self.index_name):
            print('[OK] index:', self.index_name)
            if self.check_exist_doc_type(self.index_name, self.doc_type_name):
                print('[OK] doc type:', self.doc_type_name)
                self.es_status = True
            else:
                print('[WARN] not found doc type: %s in index: %s' % \
                      (self.index_name, self.doc_type_name))
        else:
            print('[WARN] not found index:', self.index_name)

        if self.es_status:
            print('Enjoy query!')
        else:
            print('Please load data to es from file or textlist!')
        print('==========')

    def check_exist_index(self, index_name):
        return self.es.indices.exists(index=index_name)

    def check_exist_doc_type(self, index_name, doc_type_name):
        return self.es.indices.exists_type(index=index_name, doc_type=doc_type_name)

    def set_mapping(self):
        # 对于原问题录入，不做任何索引，不可作索引查询
        # 对于原问题切分好的词，指定ES分词器为simple，只按空格切分，且大写转小写
        my_mapping = {"mappings":
            {self.doc_type_name: {
                "properties": {
                    "context": {"type": "text", "index": "false"},  # 原问题不作任何索引
                    "splwords": {"type": "text", "index": "false"},  # 分词结果
                    "keywords": {"type": "text", "analyzer": "simple"},  # 抽取出的关键词
                    "embeding": {"type": "object", "enabled": "false"},  # 句向量
                    "context_id": {"type": "text"}}
            }}
        }

        # 创建Index和mapping
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        create_index = self.es.indices.create(
            index=self.index_name, body=my_mapping)  # {u'acknowledged': True}

        if not create_index["acknowledged"]:
            print("Index data bug...")

    def make_action(self, fields, actions, id):
        # fix bug: IndexError: list index out of range
        try:
            splwords, keywords, embeding = self.split_word(fields)

        except Exception as e:  # 尝试捕捉分词模块的未知bug
            print(e)
            print('failed at:', fields)
            return
        # fix bug: 'NoneType' object has no attribute 'tolist'
        if not keywords:
            print('[Error] not found any keywords:', fields)
            return
        if embeding is None:
            return

        try:
            action = {
                "_index": self.index_name,
                "_type": self.doc_type_name,
                "_source": {
                    "context": fields,
                    "splwords": splwords,
                    "keywords": keywords,
                    "embeding": embeding.tolist(),
                    "context_id": id,
                }
            }
            actions.append(action)
            #print(actions)
        except Exception as e:
            print('fields:', fields)
            print(e)

    def load_data_from_file(self, input_file, overwrite=False):
        '''
        从文件导入数据并写入到ES数据库
        input_file: 文件名
        filed_sep: 行内域分隔符
        skip_lines: 跳过前面的行数，默认为0
        read_lines: 最大读取行数，默认为-1（无上限）
        filter_repet: 是否过滤重复问题
        overwrite: 是否覆盖已有数据库
        '''

        # 覆盖模式或者ES未准备好得状态下：重新定义mapping
        if overwrite or (not self.es_status):
            self.set_mapping()

        print("Indexing data.....")

        ACTION = []  # 创建ACTIONS
        df = pd.read_csv(input_file)
        for i in range(len(df)):
            self.make_action(df["knowledge_q"][i], ACTION, i)
            self.questions_hash.add(i)

            # 批量导入
        success, _ = bulk(
            self.es, ACTION, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)
        print("Index data success")
        self.check_es_status()  # 再次检查ES状态
    # 求句子向量用于
    def sentence2vec(self,query):
        sentence = []
        for word in query:
            if word in self.word2vec:
                sentence.append(self.word2vec[word])
        sentence = np.array(sentence)
        if len(sentence) > 0:
            sentence = sentence.sum(axis=0) / len(sentence)  # 计算句向量
        else:
            sentence = np.zeros(shape=(300, ))
        return sentence

    def split_word(self, query):
        '''
        jieba分词器：返回分词、关键词
        '''
        cut =jieba.lcut(query, cut_all=True)
        splwords = "/".join(jieba.cut(query))
        embeding = self.sentence2vec(cut)
        keywords = " ".join(jieba.analyse.extract_tags(query,topK=8))
        return splwords,keywords,embeding

    @classmethod
    def softmax(cls, x, inverse=False):
        if inverse:
            ex = np.exp(-np.array(x))
        else:
            ex = np.exp(x)
        return ex / ex.sum()

    def set_es_weight(self, es_weight):
        '''
        调整es和bert各自贡献程度的超参数，两者之和固定是为1
        '''
        self.es_weight = es_weight
        self.bert_weight = 1 - es_weight

    def calc_blue(self, sent1, sent2):
        '''
        计算BLUE
        sent1：真实句子序列
        sent2：预测输出序列
        '''
        sent1_list = [w for w in sent1.split('/') \
                      if w not in self.punctuation]  # 去除单词中的标点符号
        sent2_list = [w for w in sent2.split('/') \
                      if w not in self.punctuation]  # 去除单词中的标点符号
        sent1_len = len(sent1_list)
        sent2_len = len(sent2_list)

        # 计算惩罚因子BP
        if sent2_len >= sent1_len:
            BP = 1
        else:
            BP = np.exp(1 - sent1_len / sent2_len)

        # 计算输出预测精度p
        pv = [min(sent1_list.count(w), sent2_list.count(w)) \
              for w in set(sent2_list)]

        return BP * np.log(sum(pv) / sent2_len)

    def calc_similarity(self, embed1, embed2):
        '''
        计算两个句向量的余弦相似度Similarity
        '''
        num =np.dot(embed1,embed2)
        s = (np.linalg.norm(embed1)*(np.linalg.norm(embed2)))
        if s == 0:
            similar = 0.0
        else:
            similar=num/s

        return similar

    def calc_distance(self, embed1, embed2):
        '''
        计算两个句向量的欧式距离Distance
        '''
        return np.linalg.norm(embed1 - embed2)

    def query(self, sentences, n=100):
        cut = jieba.lcut(sentences, cut_all=True)  #分词
        splwords = "/".join(jieba.cut(sentences))
        keywords = jieba.analyse.extract_tags(sentences,topK=8)
        keywords = " ".join(keywords)
        embeding = self.sentence2vec(cut)

        print('query:', sentences)
        print('splwords:', splwords)
        print('keywords:', keywords)
        print()

        query_words = ' '.join([word for word in keywords.split(' ') \
                                if word not in self.stop_list])
        # 这里匹配检索只采用了keywords
        query = {'query': {"match": {"keywords": query_words}}}
        res = self.es.search(self.index_name, self.doc_type_name, body=query)
        hit_nums = res['hits']['total']
        print('hit number: %d, set reutrn: %d' % (hit_nums, n))
        hits = res['hits']['hits']


        blues = []
        similars =[]
        for hit in hits:
            embed = np.array(hit['_source']['embeding'])
            splword = hit['_source']['splwords']
            blue = self.calc_blue(splwords, splword)
            similar = self.calc_similarity(embed, embeding)
            blues.append(blue)
            similars.append(similar)

        scores = np.array(blues)+ np.array(similars)
        scores_idx = np.argsort(scores)[::-1]

        context_id = []
        context = []
        score = []
        answers =[]
        for idx in scores_idx[:n]:
            context_id.append(hits[idx]['_source']['context_id'])
            answers.append(pd.read_csv(TRAIN_PATH)["knowledge_a"][hits[idx]['_source']['context_id']])
            context.append(hits[idx]['_source']['context'])
            score.append(scores[idx])
        for i in range(5):
            print('------------------------------')
            print('context_id:',list(set(context_id))[i])
            print('context:', list(set(context))[i])
            print('answers:', list(set(answers))[i])
            print('score:', list(set(score))[i])


            # print('context_id:', hits[idx]['_source']['context_id'])
            # print('context:', hits[idx]['_source']['context'])
            # print('answers:', pd.read_csv(TRAIN_PATH)["knowledge_a"][hits[idx]['_source']['context_id']])
            # print('keywords:', hits[idx]['_source']['keywords'])
            # print('score:', scores[idx])
            # print('blue:', blues[idx])
            # print('similarity:', similars[idx])
        return context_id, context, score

if __name__ == "__main__":
    index_name='jinrong'
    doc_type_name='text'
    esearch = ESearch(index_name, doc_type_name)

    # 测试esearch
    esearch.load_data_from_file(TRAIN_PATH)

    # 查询
    print()
    esearch.query('房贷利率是用等额本金还是等额本息好啊？')
    #进行理财什么平台比较好呢？
    #贷款公司哪家好?
    #夫妻使用公积金贷款买房需要满足哪些条件？
    #esearch.query('夫妻婚前财产怎么划分呢？')
    #
    #esearch.query('贷款买房需要满足什么？')
